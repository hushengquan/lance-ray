import logging
from typing import Any, Optional

import lance
from lance.lance import CompactionMetrics
from lance.optimize import Compaction, CompactionOptions, CompactionTask

from .index import _map_async_with_pool
from .utils import (
    array_split,
    create_storage_options_provider,
    get_or_create_namespace,
    validate_uri_or_namespace,
)

logger = logging.getLogger(__name__)


def _handle_compaction_task(
    dataset_uri: str,
    storage_options: Optional[dict[str, str]] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    table_id: Optional[list[str]] = None,
):
    """
    Create a function to handle a batch of compaction tasks for use with Pool.
    The returned callable opens the dataset once per worker invocation and
    executes all tasks in the batch sequentially, avoiding repeated dataset
    open overhead.
    """

    def func(tasks: list[CompactionTask]) -> dict[str, Any]:
        """
        Execute a batch of compaction tasks on a single dataset connection.

        Args:
            tasks: List of CompactionTask objects to execute.

        Returns:
            Dictionary with status and result information.
        """
        # Create storage options provider in worker for credentials refresh
        storage_options_provider = create_storage_options_provider(
            namespace_impl, namespace_properties, table_id
        )

        # Open dataset once for the entire batch
        dataset = lance.LanceDataset(
            dataset_uri,
            storage_options=storage_options,
            storage_options_provider=storage_options_provider,
        )

        results = []
        errors = []
        for task in tasks:
            try:
                logger.info(
                    "Executing compaction task for fragments %s", task.fragments
                )
                result = task.execute(dataset)
                logger.info(
                    "Compaction task completed for fragments %s", task.fragments
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    "Compaction task failed for fragments %s: %s",
                    task.fragments,
                    e,
                )
                errors.append(
                    {"fragments": task.fragments, "error": str(e)}
                )

        if errors:
            return {
                "status": "error",
                "results": results,
                "errors": errors,
            }

        return {
            "status": "success",
            "results": results,
        }

    return func


def compact_files(
    uri: Optional[str] = None,
    *,
    table_id: Optional[list[str]] = None,
    compaction_options: Optional[CompactionOptions] = None,
    num_workers: int = 4,
    storage_options: Optional[dict[str, str]] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    ray_remote_args: Optional[dict[str, Any]] = None,
) -> Optional[CompactionMetrics]:
    """
    Compact files in a Lance dataset using distributed Ray workers.

    This function distributes the compaction process across multiple Ray workers,
    with each worker executing a subset of compaction tasks. The results are then
    committed as a single compaction operation.

    Args:
        uri: The URI of the Lance dataset to compact. Either uri OR
            (namespace_impl + table_id) must be provided.
        table_id: The table identifier as a list of strings. Must be provided
            together with namespace_impl.
        compaction_options: Options for the compaction operation.
        num_workers: Number of Ray workers to use (default: 4).
        storage_options: Storage options for the dataset.
        namespace_impl: The namespace implementation type (e.g., "rest", "dir").
            Used together with table_id for resolving the dataset location and
            credentials vending in distributed workers.
        namespace_properties: Properties for connecting to the namespace.
            Used together with namespace_impl and table_id.
        ray_remote_args: Options for Ray tasks (e.g., num_cpus, resources).

    Returns:
        CompactionMetrics with statistics from the compaction operation.

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If compaction fails.
    """
    validate_uri_or_namespace(uri, namespace_impl, table_id)

    merged_storage_options: dict[str, Any] = {}
    if storage_options:
        merged_storage_options.update(storage_options)

    # Resolve URI and get storage options from namespace if provided
    namespace = get_or_create_namespace(namespace_impl, namespace_properties)
    if namespace is not None and table_id is not None:
        from lance_namespace import DescribeTableRequest

        describe_response = namespace.describe_table(DescribeTableRequest(id=table_id))
        uri = describe_response.location
        if describe_response.storage_options:
            merged_storage_options.update(describe_response.storage_options)

    # Create storage options provider for local operations
    storage_options_provider = create_storage_options_provider(
        namespace_impl, namespace_properties, table_id
    )

    # Load dataset
    dataset = lance.LanceDataset(
        uri,
        storage_options=merged_storage_options,
        storage_options_provider=storage_options_provider,
    )

    logger.info("Starting distributed compaction")

    # Step 1: Create the compaction plan
    compaction_plan = Compaction.plan(dataset, compaction_options)

    logger.info(f"Compaction plan created with {compaction_plan.num_tasks()} tasks")

    if compaction_plan.num_tasks() == 0:
        logger.info("No compaction tasks needed")
        return None

    # Adjust num_workers if needed
    num_tasks = compaction_plan.num_tasks()
    if num_workers > num_tasks:
        num_workers = num_tasks
        logger.info("Adjusted num_workers to %d to match task count", num_workers)

    # Split tasks into batches so each worker processes multiple tasks on
    # a single dataset connection, reducing repeated dataset open overhead
    task_batches = array_split(compaction_plan.tasks, num_workers)

    # Step 2: Execute task batches in parallel using Ray Pool
    task_handler = _handle_compaction_task(
        dataset_uri=uri,
        storage_options=merged_storage_options,
        namespace_impl=namespace_impl,
        namespace_properties=namespace_properties,
        table_id=table_id,
    )

    results = _map_async_with_pool(
        fragment_handler=task_handler,
        fragment_batches=task_batches,
        num_workers=num_workers,
        ray_remote_args=ray_remote_args,
        error_prefix="Failed to complete distributed compaction",
    )

    # Check for failures
    failed_results = [r for r in results if r["status"] == "error"]
    if failed_results:
        error_messages = []
        for r in failed_results:
            for err in r["errors"]:
                error_messages.append(
                    f"fragments {err['fragments']}: {err['error']}"
                )
        raise RuntimeError(f"Compaction failed: {'; '.join(error_messages)}")

    # Step 3: Collect successful RewriteResult objects from all batches
    rewrites = []
    for r in results:
        rewrites.extend(r["results"])

    logger.info(
        f"Collected {len(rewrites)} successful compaction results, committing..."
    )

    # Step 4: Commit the compaction
    metrics = Compaction.commit(dataset, rewrites)

    logger.info(f"Compaction completed successfully. Metrics: {metrics}")

    return metrics


def compact_database(
    *,
    database: list[str],
    namespace_impl: str,
    namespace_properties: Optional[dict[str, str]] = None,
    compaction_options: Optional[CompactionOptions] = None,
    num_workers: int = 4,
    storage_options: Optional[dict[str, str]] = None,
    ray_remote_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Compact all tables under a given database (namespace) using distributed Ray workers.

    This function lists all tables under the specified database via the namespace API,
    then runs :func:`compact_files` on each table. Use this when you want to compact
    an entire database instead of a single table.

    Args:
        database: The database (namespace) identifier as a list of path segments,
            e.g. ``["my_database"]``. All tables under this namespace will be compacted.
        namespace_impl: The namespace implementation type (e.g. ``"rest"``, ``"dir"``).
            Required for resolving table locations and credentials.
        namespace_properties: Properties for connecting to the namespace.
        compaction_options: Options for the compaction operation (used for every table).
        num_workers: Number of Ray workers per table (default: 4).
        storage_options: Storage options for the datasets.
        ray_remote_args: Options for Ray tasks (e.g. num_cpus, resources).

    Returns:
        A list of dicts, one per table, with keys:
        - ``"table_id"``: ``list[str]`` – full table identifier (database + table name).
        - ``"metrics"``: :class:`~lance.lance.CompactionMetrics` or ``None`` –
          compaction result for that table, or ``None`` if no compaction was needed.

    Raises:
        ValueError: If database is empty or namespace_impl is not provided.
        RuntimeError: If listing tables fails or any table compaction fails.

    Example:
        >>> results = compact_database(
        ...     database=["my_db"],
        ...     namespace_impl="dir",
        ...     namespace_properties={"root": "/path/to/tables"},
        ...     compaction_options=CompactionOptions(target_rows_per_fragment=10000),
        ...     num_workers=2,
        ... )
        >>> for item in results:
        ...     print(item["table_id"], item["metrics"])
    """
    if not database:
        raise ValueError("'database' must be a non-empty list of path segments.")
    if not namespace_impl:
        raise ValueError(
            "'namespace_impl' is required when using compact_database."
        )

    from lance_namespace import ListTablesRequest

    namespace = get_or_create_namespace(namespace_impl, namespace_properties)
    if namespace is None:
        raise RuntimeError(
            "Failed to create namespace from namespace_impl and namespace_properties."
        )

    # List all tables under the database (namespace) with pagination
    all_tables: list[str] = []
    page_token: Optional[str] = None
    limit = 500

    while True:
        request = ListTablesRequest(
            id=database,
            page_token=page_token,
            limit=limit,
        )
        response = namespace.list_tables(request)
        all_tables.extend(response.tables)
        page_token = getattr(response, "page_token", None)
        if not page_token:
            break

    if not all_tables:
        logger.info("No tables found under database %s, nothing to compact.", database)
        return []

    # table_id = database + [table_name] for each table under this namespace
    table_ids = [database + [t] for t in all_tables]
    results: list[dict[str, Any]] = []

    for table_id in table_ids:
        logger.info("Compacting table %s", table_id)
        try:
            metrics = compact_files(
                uri=None,
                table_id=table_id,
                compaction_options=compaction_options,
                num_workers=num_workers,
                storage_options=storage_options,
                namespace_impl=namespace_impl,
                namespace_properties=namespace_properties,
                ray_remote_args=ray_remote_args,
            )
            results.append({"table_id": table_id, "metrics": metrics})
        except Exception as e:
            logger.exception("Compaction failed for table %s: %s", table_id, e)
            raise RuntimeError(
                f"Compaction failed for table {table_id}: {e}"
            ) from e

    return results
