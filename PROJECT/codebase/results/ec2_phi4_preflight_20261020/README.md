# EC2 Phi-4 preflight — 20 October 2026

This folder contains a small EC2 test and its lifecycle checks. No full EC2
experiment was launched. The existing Kaggle full run was left untouched;
Phi-4 had already started there after LLaMA finished.

## GPU and environment

- EC2: `i-03f96c0b0edb6326a`, `g4dn.xlarge`, `us-east-1`.
- NVIDIA Tesla T4, 15,360 MiB; driver `580.178.04`.
- Ubuntu 24.04, kernel `7.0.0-1014-aws`; 50 GiB EBS root disk.
- About 35 GiB free after installing the environment and model.
- Python 3.12, PyTorch `2.11.0+cu128`, Transformers `5.17.0`,
  Accelerate `1.15.0`, bitsandbytes `0.50.2`.
- Model: `unsloth/Phi-4-mini-instruct-bnb-4bit`, pinned revision
  `cece1fd36f04ff79f55ec861f206ca4e16acea6e`.

## Actual inference test

The first two fixed MedQA IDs and the first two fixed MATH IDs were used,
without selecting questions based on fresh outputs. There were four
conversations, twelve answer turns, and eight optional tool opportunities.
The original P0/P1/P2 prompts, 160-token answer budget, separate 1/2 action
selection, and one exact-ID question–answer entry were preserved.

Phi-4 made all eight calls. All eight matching reference entries were checked
in the actual generation inputs; P1 evidence remained in P2 history. MATH
entries retained the original solutions. The helper verified that reopening a
completed checkpoint generated no additional answers. The parser was not
changed and scoring remains provisional.

## Lifecycle checks

| Test | Evidence |
|---|---|
| GPU generation | `downloaded/runs/phi4-smoke/completed.json` |
| Normal exit | Success status and checkpoint saved |
| Intentional Python error | `exit-code` status and checkpoint saved |
| Hung process | Killed at the 3-second test limit; `timeout` status saved |
| Independent timer | Separate timer fired and saved `backup-deadline` |
| Finalizer ordering | Local tests verify durable status before requesting shutdown |
| Actual automatic stop | The successful GPU job requested poweroff; AWS reached Stopped without an AWS stop API call |
| Restart | AWS start command succeeded; CUDA and saved results survived |
| No rerun at boot | Experiment service and backup timer were both inactive |

The error, timeout, and short deadline probes deliberately suppressed actual
poweroff so all three could be inspected in one boot. They used the same
finalizer as the real smoke job. Actual poweroff was exercised by the real
GPU job. Shutdown took several minutes in AWS's Stopping state; no force-stop
API was used to complete the automatic-stop test.

## Installed controls

- `ethics-phi4-smoke.service`: explicitly started, no restart policy,
  maximum runtime 20 minutes, stop handler saves file hashes/status and
  requests Ubuntu poweroff on success, error, or timeout.
- `ethics-phi4-deadline.timer`: independent 25-minute backup shutdown when
  explicitly armed for this test.
- Neither unit is enabled at boot. They are test units, not a full-run launch.
- Instance shutdown behavior was verified as Stop. No AWS access keys or
  private SSH keys were copied onto EC2.
- Checkpoints, inputs, replies, tool records and logs are on EBS and have also
  been downloaded locally. Hashes verify that the files survived shutdown.

Use different runtime limits and a separate results folder for any future full
EC2 run; do not treat these four conversations as full-run results.

The submitted report, parser, full cohort/protocol, and already-submitted
Kaggle bundle are protected by `source_guards.json`. Final machine-readable
results are in `final_verification.json` and `aws_lifecycle_verification.json`.
