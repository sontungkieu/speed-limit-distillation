"""Named vLLM RPC methods for loading this campaign's tensor snapshots."""


class RefitExtension:
    def load_student_snapshot(self, snapshot):
        import torch

        weights = torch.load(snapshot, map_location="cpu", weights_only=True)
        model = self.model_runner.model
        loaded = model.load_weights(weights.items())
        torch.cuda.synchronize()
        probes = {}
        for name in ["model.norm.weight", "model.embed_tokens.weight"]:
            value = model.get_parameter(name).detach().flatten()
            probes[name] = value[:16].float().cpu().tolist()
        return {"loaded_tensors": len(loaded), "probes": probes}
