from types import SimpleNamespace

import torch

from greek_char_bert.modelling.prediction_head import CharMLMHead


def test_formatted_preds_include_top_k_predictions():
    head = CharMLMHead()
    label_map = {0: "[CLS]", 1: "[MASK]", 2: "_", 3: "[SEP]", 4: "α", 5: "β"}
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 3.0],
                [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            ]
        ]
    )
    input_ids = torch.tensor([[0, 1, 2, 3]])
    lm_label_ids = torch.tensor([[-1, 1, 2, -1]])
    padding_mask = torch.tensor([[1, 1, 1, 1]])
    samples = [SimpleNamespace(clear_text={"doc": ["[MASK]", "_"]})]

    preds = head.formatted_preds(
        logits,
        label_map,
        samples,
        top_k=2,
        input_ids=input_ids,
        lm_label_ids=lm_label_ids,
        padding_mask=padding_mask,
    )

    prediction = preds[0]["predictions"]
    assert prediction["predictions"] == ["α"]
    assert prediction["text_with_preds"] == "[α]_"
    assert prediction["text_with_top_k_preds"] == "[α/β]_"
    assert prediction["top_k_predictions"][0][0]["token"] == "α"
    assert prediction["top_k_predictions"][0][1]["token"] == "β"
