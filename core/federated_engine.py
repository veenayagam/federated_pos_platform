import torch
import torch.nn as nn
from opacus import PrivacyEngine
import flwr as fl
from flwr.client import NumPyClient
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from core.model import MenuUpsellModel
from core.data_processor import process_dynamic_csv, build_global_market_testloader

GLOBAL_TESTLOADER = build_global_market_testloader()

def evaluate_on_global_market(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for a, t, y in GLOBAL_TESTLOADER:
            preds = model(a, t)
            correct += ((preds > 0.5).float() == y).sum().item()
            total += y.size(0)
    return correct / total

class CloverDPClient(NumPyClient):
    def __init__(self, rest_id, dataloader, step_history):
        self.rest_id = rest_id
        self.net = MenuUpsellModel()
        self.dataloader = dataloader
        self.step_history = step_history
        self.epochs = 3

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.net.parameters()]

    def set_parameters(self, parameters):
        for param, new_val in zip(self.net.parameters(), parameters):
            param.data = torch.tensor(new_val, dtype=param.dtype)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.02)
        criterion = nn.BCELoss()

        privacy_engine = PrivacyEngine()
        model_dp, optimizer_dp, loader_dp = privacy_engine.make_private(
            module=self.net, optimizer=optimizer, data_loader=self.dataloader,
            noise_multiplier=1.15, max_grad_norm=1.0,
        )

        sample_rate = 1 / len(loader_dp)
        for _ in range(self.step_history.get(self.rest_id, 0)):
            privacy_engine.accountant.step(noise_multiplier=1.15, sample_rate=sample_rate)

        samples, correct = 0, 0
        for _ in range(self.epochs):
            for a, t, y in loader_dp:
                optimizer_dp.zero_grad()
                preds = model_dp(a, t)
                loss = criterion(preds, y)
                loss.backward()
                optimizer_dp.step()
                self.step_history[self.rest_id] = self.step_history.get(self.rest_id, 0) + 1
                correct += ((preds > 0.5).float() == y).sum().item()
                samples += y.size(0)

        eps = privacy_engine.get_epsilon(delta=1e-4)
        return self.get_parameters({}), len(self.dataloader.dataset), {
            "train_acc": float(correct / max(samples, 1)),
            "epsilon": float(eps),
        }

def get_eval_fn():
    # Centralized evaluation directly on the server process
    def evaluate(server_round, parameters, config):
        net = MenuUpsellModel()
        for param, new_val in zip(net.parameters(), parameters):
            param.data = torch.tensor(new_val, dtype=param.dtype)
        acc = evaluate_on_global_market(net)
        return 0.0, {"acc": acc}
    return evaluate

def run_federated_orchestration(uploaded_files, num_rounds=10):
    num_clients = len(uploaded_files)
    step_history = {i: 0 for i in range(num_clients)}

    def client_fn(cid: str):
        c_idx = int(cid)
        uploaded_files[c_idx].seek(0)
        loader = process_dynamic_csv(uploaded_files[c_idx], seed=c_idx)
        return CloverDPClient(c_idx, loader, step_history)

    def fit_metrics_agg(metrics):
        if not metrics: return {}
        max_eps = max(m["epsilon"] for _, m in metrics)
        return {"epsilon": max_eps}

    strategy = FedAvg(
        fraction_fit=1.0, 
        min_fit_clients=num_clients, 
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=fit_metrics_agg, 
        evaluate_fn=get_eval_fn(),
    )

    # Capture the History object securely bridging the background Ray process
    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
# Extract tuples formatted as: [(round_number, metric_value), ...]
    eps_tuples = history.metrics_distributed_fit.get("epsilon", [])
    acc_tuples = history.metrics_centralized.get("acc", [])

    # Isolate just the y-axis values, STRICTLY ignoring Round 0 to align lengths
    eps_values = [val for rnd, val in eps_tuples if rnd > 0]
    acc_values = [val for rnd, val in acc_tuples if rnd > 0]

    # Safety fallback to prevent UI crash if a round skips
    if not eps_values: eps_values = [0.0] * num_rounds
    if not acc_values: acc_values = [0.0] * num_rounds

    return {"acc": acc_values, "epsilon": eps_values}