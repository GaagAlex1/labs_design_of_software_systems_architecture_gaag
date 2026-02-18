import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModel


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_state[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    bs = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(bs, device=last_hidden_state.device), seq_lens]


class TritonPythonModel:
    def initialize(self, args):
        model_id = "Qwen/Qwen3-Embedding-0.6B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)

        self.max_length = 8192

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            arr = in_tensor.as_numpy()  # dtype=object (bytes)
            texts = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr.reshape(-1)]

            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                out = self.model(**batch)
                emb = _last_token_pool(out.last_hidden_state, batch["attention_mask"])
                emb = F.normalize(emb, p=2, dim=1)

            emb_np = emb.detach().cpu().to(torch.float32).numpy()
            out_tensor = pb_utils.Tensor("EMBEDDING", emb_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses