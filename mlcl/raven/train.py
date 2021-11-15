import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from mlcl.config import Config
from mlcl.raven.data.data_module import RavenDataModule
from mlcl.raven.module.raven_module import RavenModule


@hydra.main(config_path='../config')
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    print(cfg)
    pl.seed_everything(cfg.seed)
    module: RavenModule = instantiate(cfg.mlcl.raven.module, cfg)
    data_module = RavenDataModule(cfg)

    trainer: pl.Trainer = instantiate(cfg.pytorch_lightning.trainer)
    trainer.fit(module, data_module)
    trainer.test(module, data_module)

    hparams = {'hp': OmegaConf.to_container(cfg, resolve=True)}
    hparams['hp/model/params_total'] = sum(p.numel() for p in module.parameters())
    hparams['hp/model/params_trainable'] = sum(p.numel() for p in module.parameters() if p.requires_grad)
    hparams['hp/model/params_not_trainable'] = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    trainer.logger.log_hyperparams(hparams)
    trainer.logger.finalize('success')


if __name__ == '__main__':
    load_dotenv()
    Config()
    main()
