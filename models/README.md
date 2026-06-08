# Model Artifacts

This directory holds metadata for trained Obadh models. Large generated artifacts
are ignored by git by default:

```text
*.onnx
*.ort
*.pt
*.pth
*.mlmodel
*.mlpackage
```

Each checked-in model release directory should contain small metadata files such
as:

```text
config.json
vocab.input.json
vocab.output.json
metrics.json
LICENSE_DATA.md
```

Do not add a model artifact until it has measured latency, memory, accuracy, and
fallback behavior on target mobile hardware.
