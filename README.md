<img src="./extra/vision-logo.png" alt="Vision Chess Logo" width=300 style="display:block; margin: auto; width: 50%"></img>

# About
Vision is a small in-progress model designed to predict the next move in a chess game using a GPT-like model.

It's a hobby project helping me learn more about machine learning without using/referencing python-nnue.

# Installation
Installation can be done with poetry:

```bash
poetry install
```

# Usage
1. You'll need to get some pgn files and place them in a folder (ex. `data/pgn`)

    ```bash
    mkdir -p data/pgn && mkdir -p data/training && mkdir -p data/validation
    ```
    (Note: You can place the pgn files where you want but the next step expects there to be a `training` and `validation` directory within the folder you specify)
2. Then process these into numpy arrays using `pgn_to_npy.py`:

    ```bash
    $(poetry env activate)
    python vision/pgn_to_npy.py --input ./data/pgn --output-dir ./data
    ```
    (Note: The search for PGN files isn't recursive, it will only look at the top level of the directory. There is also no progress bar output just yet.)
    

3. Then run the model, you can also tweak `vision/model/config.py`

    ```bash
    python vision/main.py
    ```

4. The model will run and automatically save any epochs that perform better than the last best epoch. You should see output like:

    ```bash
    Training on approximately 9963 batches.
    Validating on approximately 1109 batches.
    Batch size: 4
    --- Epoch 0 ---
    Training Epoch: 0 | Batch: 0 | Sample input: tensor([1674, 2942], device='mps:0') | Running Loss: 8.81575 | Running Perplexity: 6739.52930
    Training Epoch: 0 | Batch: 10 | Sample input: tensor([1739, 2291], device='mps:0') | Running Loss: 5.41107 | Running Perplexity: 223.87140
    Training Epoch: 0 | Batch: 20 | Sample input: tensor([1804, 2738], device='mps:0') | Running Loss: 4.52343 | Running Perplexity: 92.15147
    Training Epoch: 0 | Batch: 30 | Sample input: tensor([1804, 2226], device='mps:0') | Running Loss: 3.94177 | Running Perplexity: 51.50982
    Training Epoch: 0 | Batch: 40 | Sample input: tensor([1804, 2356], device='mps:0') | Running Loss: 3.67486 | Running Perplexity: 39.44297
    ```

# Caveats / Limitations

The current embedding scheme is based on UCI move instructions from the start of a standard chess game. As such it likely won't be any good at playing Chess variants like Chess960. This embedding scheme also makes dropout ineffective. It may change in the future as the model evolves.