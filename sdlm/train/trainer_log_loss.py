import os
import torch
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer import unwrap_model, _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import Trainer
import torch.distributed as dist


class CustomTrainerForLogloss(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.micro_steps = 0  

        self.micro_loss1 = 0.0
        self.micro_loss2 = 0.0
        self.micro_loss3 = 0.0
        self.micro_loss4 = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs, extra_loss = model(**inputs)
        # print(f'{extra_loss=}')

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if len(extra_loss) > 4:
            extra_loss = extra_loss[:4]  # only 4 
        loss_pos_1, loss_pos_2, loss_pos_3, loss_pos_4 = extra_loss

        # TODO  a + nan = nan
        if not torch.isnan(loss_pos_1):
            self.micro_loss1 += loss_pos_1.detach()
        if not torch.isnan(loss_pos_2):
            self.micro_loss2 += loss_pos_2.detach()
        if not torch.isnan(loss_pos_3):
            self.micro_loss3 += loss_pos_3.detach()
        if not torch.isnan(loss_pos_4):
            self.micro_loss4 += loss_pos_4.detach()
        self.micro_steps += 1

        if self.micro_steps % self.args.gradient_accumulation_steps == 0:
            avg_loss1 = self.micro_loss1 / self.args.gradient_accumulation_steps
            avg_loss2 = self.micro_loss2 / self.args.gradient_accumulation_steps
            avg_loss3 = self.micro_loss3 / self.args.gradient_accumulation_steps
            avg_loss4 = self.micro_loss4 / self.args.gradient_accumulation_steps

            if self.args.local_rank != -1:
                avg_loss1 = torch.tensor(0.0).to(self.args.device) if type(avg_loss1) == float else avg_loss1.to(self.args.device)
                avg_loss2 = torch.tensor(0.0).to(self.args.device) if type(avg_loss2) == float else avg_loss2.to(self.args.device)
                avg_loss3 = torch.tensor(0.0).to(self.args.device) if type(avg_loss3) == float else avg_loss3.to(self.args.device)
                avg_loss4 = torch.tensor(0.0).to(self.args.device) if type(avg_loss4) == float else avg_loss4.to(self.args.device)
                    
                # if type(avg_loss1) == float:
                #     avg_loss1 = torch.tensor(0.0).to(self.args.device)
                # else:
                #     avg_loss1 = avg_loss1.to(self.args.device)
                
                # if type(avg_loss2) == float:
                #     avg_loss2 = torch.tensor(0.0).to(self.args.device) if type(avg_loss2) == float else avg_loss2.to(self.args.device)
                # else:
                #     avg_loss2 = avg_loss2.to(self.args.device)

                # if type(avg_loss3) == float:
                #     avg_loss3 = torch.tensor(0.0).to(self.args.device)
                # else:
                #     avg_loss3 = avg_loss3.to(self.args.device)
                
                # if type(avg_loss4) == float:
                #     avg_loss4 = torch.tensor(0.0).to(self.args.device)
                # else:
                #     avg_loss4 = avg_loss4.to(self.args.device)

                dist.all_reduce(avg_loss1, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss2, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss3, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss4, op=dist.ReduceOp.SUM)

                avg_loss1 /= self.args.world_size
                avg_loss2 /= self.args.world_size
                avg_loss3 /= self.args.world_size
                avg_loss4 /= self.args.world_size

            self.state.current_losses = {
                "loss_pos_1": avg_loss1.item(),
                "loss_pos_2": avg_loss2.item(),
                "loss_pos_3": avg_loss3.item(),
                "loss_pos_4": avg_loss4.item()
            }

            # reset 
            self.micro_loss1 = 0.0
            self.micro_loss2 = 0.0
            self.micro_loss3 = 0.0
            self.micro_loss4 = 0.0
            self.micro_steps = 0

        return (loss, outputs) if return_outputs else loss


class TensorBoardLoggingCallback(TrainerCallback):
    def __init__(self):
        self._logging_dir = None
        self.writer = None
        self._is_main_process = None  

    def _lazy_init(self, args):
        if self._logging_dir is None:
            self._logging_dir = args.logging_dir

            self._is_main_process = not dist.is_initialized() or (dist.get_rank() == 0)

            if self._is_main_process:
                os.makedirs(self._logging_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=self._logging_dir)
                print(f"TensorBoard logs will be saved to: {self._logging_dir}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if hasattr(state, "current_losses"):
            if logs is not None:
                logs.update(state.current_losses)

            self._lazy_init(args)
            if self._is_main_process and self.writer:
                step = state.global_step
                for k, v in state.current_losses.items():
                    self.writer.add_scalar(f"pos/{k}", v, step)

            del state.current_losses

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.close()

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
