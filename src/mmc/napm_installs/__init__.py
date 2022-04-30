import napm
from loguru import logger


def napm_pi_katcloob():
    """
    Usage:

        import cloob
        from cloob.cloob_training import model_pt, pretrained

        config = pretrained.get_config('cloob_laion_400m_vit_b_16_16_epochs')
        model = model_pt.get_pt_model(config)
        checkpoint = pretrained.download_checkpoint(config)
        model.load_state_dict(model_pt.get_pt_params(config, checkpoint), )
        model.eval().requires_grad_(False).to('cuda')
    """
    logger.debug('using napm to "install" katCLOOB')
    url = "https://github.com/crowsonkb/cloob-training"
    napm.pseudoinstall_git_repo(url, package_name='cloob')


def all():
    napm_pi_katcloob()


if __name__ == '__main__':
    all()