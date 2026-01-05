"""
Download LightVLA checkpoints from HuggingFace.

These are the token-pruning optimized versions built on top of OpenVLA-OFT.
"""

from huggingface_hub import snapshot_download

models = [
    'TTJiang/LightVLA-libero-spatial',
    'TTJiang/LightVLA-libero-object',
    'TTJiang/LightVLA-libero-goal',
    'TTJiang/LightVLA-libero-10'
]

for model in models:
    print(f'Downloading {model}...')
    snapshot_download(repo_id=model)
    print(f'✓ Completed {model}')

print('\n✅ All LightVLA checkpoints downloaded successfully!')
