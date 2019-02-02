import click

from . import api, util


@click.command()
@click.argument("fin", type=click.Path(exists=True), metavar='<path to input file>')
@click.argument("fout", type=click.Path(), metavar='<path to output file>')
@click.argument("n_factors", type=click.INT, metavar='<number of factors>')
@click.option("--max_iter", type=click.INT, default=20, help="Maximum number of iterations")
@click.option("--min_iter", type=click.INT, default=5, help="Minimum number of iterations")
def cli(fin, fout, n_factors, max_iter, min_iter):
    """variational Latent Gaussian Process (vLGP)"""
    click.echo("Loading {}".format(fin))
    trials = util.load(fin)
    click.secho("{} loaded".format(fin), fg="green")

    result = api.fit(trials, n_factors, max_iter=max_iter, min_iter=min_iter, path=fout)

    click.echo("Saving {}".format(fout))
    util.save(result, fout)
    click.secho("{} saved".format(fout), fg="green")


if __name__ == "__main__":
    cli()
