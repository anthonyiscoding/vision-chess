from dataclasses import asdict
import json
import logging
import torch
from datetime import datetime
from dynaconf import loaders
from dynaconf.utils.boxing import DynaBox
from vision.model.tokenizer import special_tokens_to_embeddings
import optuna.exceptions


logger = logging.getLogger(__name__)


def train(
    model,
    device,
    training_dataset,
    validation_dataset,
    training_dataloader,
    validation_dataloader,
    config,
    trial=None,
):
    model.to(device)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # TODO: Disabling the scheduler temporarily during testing
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)

    training_batch_count = len(training_dataset) // config.batch_size
    validation_batch_count = len(validation_dataset) // config.batch_size

    # It's probably better to have validation be the same batch limit for testing
    # TODO: But only for testing and small training sizes, otherwise it's unnecessary
    if config.batch_limit:
        training_batch_count = config.batch_limit
        validation_batch_count = config.batch_limit

    logger.info(
        "Training on approximately %d batches.",
        training_batch_count,
    )
    logger.info(
        "Validating on approximately %d batches.",
        validation_batch_count,
    )
    logger.info("Batch size: %d", config.batch_size)

    logger.info("Full config: %s", config.as_dict())

    best_loss = torch.inf
    for epoch in range(config.num_epochs):
        logger.info("Starting epoch %d", epoch)
        running_loss, running_perplexity = _training_loop(
            model,
            device,
            training_dataloader,
            config,
            loss_fn,
            optimizer,
            epoch,
        )

        val_running_loss, val_running_perplexity, accuracy = _validation_loop(
            model, device, validation_dataloader, config, loss_fn, epoch, trial=trial
        )

        logger.info(
            "Epoch %d: Avg Loss = %.5f | Avg Perplexity: %.5f | Avg Val Loss: %.5f | Avg Val Perplexity: %.5f",
            epoch,
            running_loss,
            running_perplexity,
            val_running_loss,
            val_running_perplexity,
        )

        scheduler.step()

        if config.save_model and running_loss < best_loss:
            _save_model(
                model, best_loss, epoch, config, running_loss, running_perplexity
            )
            best_loss = running_loss

    return accuracy


def _training_loop(
    model, device, training_dataloader, config, loss_fn, optimizer, epoch
):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for i, (input, target) in enumerate(training_dataloader):
        input = input.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(input)
        output = output.view(-1, output.size(-1))  # (batch*seq_len, vocab_size)
        target = target.view(-1)  # (batch*seq_len,)

        mask = target != special_tokens_to_embeddings["<|pad|>"]
        if mask.shape[0] != output.shape[0]:
            logger.warning(
                "Skipping training batch %d due to shape mismatch: mask %s, output %s",
                i,
                mask.shape,
                output.shape,
            )
            continue
        if mask.any():
            loss = loss_fn(output[mask], target[mask])
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    "Loss is %s! Skipping batch. Input: %s, Output: %s, Target: %s",
                    loss,
                    input,
                    output,
                    target,
                )
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            mask_sum = mask.sum().item()
            if mask_sum == 0:
                logger.warning(
                    "All tokens are padding, skipping batch %d with input: %s",
                    i,
                    input[0],
                )
            total_loss += loss.item() * mask_sum
            total_tokens += mask_sum
            running_loss = (
                total_loss / total_tokens if total_tokens > 0 else float("inf")
            )
            running_perplexity = torch.exp(torch.tensor(running_loss))
            if i % 10 == 0:
                logger.info(
                    "Training Epoch: %d | Batch: %d | Sample input: %s | Running Loss: %.5f | Running Perplexity: %.5f",
                    epoch,
                    i,
                    input[0][:2],
                    running_loss,
                    running_perplexity,
                )

        if config.batch_limit and i + 1 > config.batch_limit:
            break
    return running_loss, running_perplexity


def _validation_loop(
    model, device, validation_dataloader, config, loss_fn, epoch, trial=None
):
    model.eval()
    val_total_loss = 0.0
    val_total_tokens = 0
    correct = 0
    accuracy = None  # Only used in trials
    for v_i, (val_input, val_target) in enumerate(validation_dataloader):
        val_input = val_input.to(device)
        val_target = val_target.to(device)
        with torch.no_grad():
            val_output = model(val_input)
            val_output = val_output.view(-1, val_output.size(-1))
            val_target = val_target.view(-1)
            if trial:
                pred = val_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(val_target.view_as(pred)).sum().item()
            val_mask = val_target != special_tokens_to_embeddings["<|pad|>"]
            if val_mask.shape[0] != val_output.shape[0]:
                logger.warning(
                    "Skipping validation batch %d due to shape mismatch: mask %s, output %s",
                    v_i,
                    val_mask.shape,
                    val_output.shape,
                )
                continue
            if val_mask.any():
                val_loss = loss_fn(val_output[val_mask], val_target[val_mask])
                if torch.isnan(val_loss) or torch.isinf(val_loss):
                    logger.error(
                        "Loss is %s! Skipping batch. Input: %s, Output: %s, Target: %s",
                        val_loss,
                        val_input,
                        val_output,
                        val_target,
                    )
                    continue
                val_mask_sum = val_mask.sum().item()
                if val_mask_sum == 0:
                    logger.warning(
                        "All tokens are padding, skipping batch %d with input: %s",
                        v_i,
                        val_input[0],
                    )
                val_total_loss += val_loss.item() * val_mask_sum
                val_total_tokens += val_mask_sum
                val_running_loss = (
                    val_total_loss / val_total_tokens
                    if val_total_tokens > 0
                    else float("inf")
                )
                val_running_perplexity = torch.exp(torch.tensor(val_running_loss))
                if v_i % 10 == 0:
                    logger.info(
                        "Validating Epoch: %d | Batch: %d | Sample input: %s | Running Loss: %.5f | Running Perplexity: %.5f",
                        epoch,
                        v_i,
                        val_input[0][:2],
                        val_running_loss,
                        val_running_perplexity,
                    )

        if config.batch_limit and v_i + 1 > config.batch_limit:
            break
    if trial:
        accuracy = correct / min(
            len(validation_dataloader.dataset), config.batch_limit * config.batch_size
        )
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_running_loss, val_running_perplexity, accuracy


def _save_model(model, best_loss, epoch, config, running_loss, running_perplexity):
    logger.info(
        "Saving model. Model had less loss than last epoch %.5f < %.5f | Perplexity: %.5f",
        running_loss,
        best_loss,
        running_perplexity,
    )
    model_name = f"models/model-{datetime.now():%Y-%m-%d-%H-%M-%S}-env-{config.env}-epoch-{epoch}-perplexity-{running_perplexity:.3f}"
    torch.save(
        model.state_dict(),
        f"{model_name}.pt",
    )
    data = config.to_dict(env=config.env)

    # Dynaconf doesn't remove all unnecessary variables
    data.pop("LOAD_DOTENV")
    data.pop("POST_HOOKS")

    loaders.write(
        f"{model_name}-config.json",
        DynaBox(data).to_dict(),
        merge=False,
        env=config.env,
    )
