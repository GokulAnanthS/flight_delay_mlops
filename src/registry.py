"""Inspect and promote registered flight-delay model versions.

src/training/train.py registers every training run as a new version of
config.MLFLOW_MODEL_NAME and aliases it 'staging' automatically. Promoting a
version to 'production' is a deliberate, separate step done here — the API
(src/api/app.py) and batch scorer (src/inference/predict.py) only ever load
whatever version currently holds the 'production' alias, so nothing reaches
real traffic just because a training run finished.
"""

import argparse
import logging

import httpx
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_client(tracking_uri: str = config.MLFLOW_TRACKING_URI) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def list_versions(client: MlflowClient | None = None) -> list[ModelVersion]:
    client = client or get_client()
    versions = client.search_model_versions(f"name='{config.MLFLOW_MODEL_NAME}'")
    return sorted(versions, key=lambda v: int(v.version), reverse=True)


def aliases_by_version(client: MlflowClient | None = None) -> dict[str, list[str]]:
    """{version: [aliases]}. `search_model_versions` doesn't populate aliases on
    the ModelVersion objects it returns, so this reads them from the
    registered model's alias dict {alias: version} instead and inverts it."""
    client = client or get_client()
    registered_model = client.get_registered_model(config.MLFLOW_MODEL_NAME)
    by_version: dict[str, list[str]] = {}
    for alias, version in registered_model.aliases.items():
        by_version.setdefault(version, []).append(alias)
    return by_version


def promote_to_production(version: int, client: MlflowClient | None = None) -> None:
    client = client or get_client()
    client.set_registered_model_alias(config.MLFLOW_MODEL_NAME, config.MLFLOW_PRODUCTION_ALIAS, version)
    logger.info("Promoted %s v%s to '%s'", config.MLFLOW_MODEL_NAME, version, config.MLFLOW_PRODUCTION_ALIAS)
    trigger_reload()

def trigger_reload(api_url: str = config.FLIGHT_DELAY_API_URL, token: str = config.ADMIN_RELOAD_TOKEN) -> None:
    """Best-effort: ask the running API to reload its model after a promotion.

    Failing here doesn't undo the promotion -- the registry alias is already
    the source of truth. A failed reload just means the API keeps serving
    whatever it already had loaded until it's reloaded or restarted some
    other way.
    """
    try:
        response = httpx.post(
            f"{api_url}/admin/reload",
            headers={"X-Admin-Token": token or ""},
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Reloaded API at %s", api_url)
    except httpx.HTTPError as exc:
        logger.warning("Could not reload API at %s: %s", api_url, exc)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List registered model versions and their current aliases")
    promote_parser = subparsers.add_parser("promote", help="Alias a version as the production model")
    promote_parser.add_argument("version", type=int)
    args = parser.parse_args()

    client = get_client()
    if args.command == "list":
        aliases = aliases_by_version(client)
        for v in list_versions(client):
            version_aliases = ", ".join(aliases.get(v.version, [])) or "(none)"
            print(f"v{v.version}  aliases=[{version_aliases}]  run_id={v.run_id}")
    elif args.command == "promote":
        promote_to_production(args.version, client)


if __name__ == "__main__":
    main()