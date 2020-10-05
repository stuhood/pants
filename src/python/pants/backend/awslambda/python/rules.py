# Copyright 2019 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import logging
import os
from dataclasses import dataclass

from pants.backend.awslambda.common.rules import AWSLambdaFieldSet, CreatedAWSLambda
from pants.backend.awslambda.python.lambdex import Lambdex
from pants.backend.awslambda.python.target_types import (
    PythonAwsLambdaHandler,
    PythonAwsLambdaRuntime,
)
from pants.backend.python.util_rules import pex_from_targets
from pants.backend.python.util_rules.pex import (
    Pex,
    PexInterpreterConstraints,
    PexPlatforms,
    PexProcess,
    PexRequest,
    PexRequirements,
    TwoStepPex,
)
from pants.backend.python.util_rules.pex_from_targets import (
    PexFromTargetsRequest,
    TwoStepPexFromTargetsRequest,
)
from pants.core.goals.package import BuiltPackage, PackageFieldSet
from pants.engine.fs import Digest, MergeDigests
from pants.engine.process import ProcessResult
from pants.engine.rules import Get, collect_rules, rule
from pants.engine.unions import UnionRule
from pants.option.global_options import GlobalOptions
from pants.util.logging import LogLevel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PythonAwsLambdaFieldSet(PackageFieldSet, AWSLambdaFieldSet):
    required_fields = (PythonAwsLambdaHandler, PythonAwsLambdaRuntime)

    handler: PythonAwsLambdaHandler
    runtime: PythonAwsLambdaRuntime


@dataclass(frozen=True)
class LambdexSetup:
    requirements_pex: Pex


@rule(desc="Create Python AWS Lambda", level=LogLevel.DEBUG)
async def create_python_awslambda(
    field_set: PythonAwsLambdaFieldSet, lambdex_setup: LambdexSetup, global_options: GlobalOptions
) -> CreatedAWSLambda:
    # Lambdas typically use the .zip suffix, so we use that instead of .pex.
    disambiguated_pex_filename = os.path.join(
        field_set.address.spec_path.replace(os.sep, "."), f"{field_set.address.target_name}.zip"
    )
    if global_options.options.pants_distdir_legacy_paths:
        pex_filename = f"{field_set.address.target_name}.zip"
        logger.warning(
            f"Writing to the legacy subpath: {pex_filename}, which may not be unique. An "
            f"upcoming version of Pants will switch to writing to the fully-qualified subpath: "
            f"{disambiguated_pex_filename}. You can effect that switch now (and silence this "
            f"warning) by setting `pants_distdir_legacy_paths = false` in the [GLOBAL] section of "
            f"pants.toml."
        )
    else:
        pex_filename = disambiguated_pex_filename
    # We hardcode the platform value to the appropriate one for each AWS Lambda runtime.
    # (Running the "hello world" lambda in the example code will report the platform, and can be
    # used to verify correctness of these platform strings.)
    py_major, py_minor = field_set.runtime.to_interpreter_version()
    platform = f"linux_x86_64-cp-{py_major}{py_minor}-cp{py_major}{py_minor}"
    # set pymalloc ABI flag - this was removed in python 3.8 https://bugs.python.org/issue36707
    if py_major <= 3 and py_minor < 8:
        platform += "m"
    if (py_major, py_minor) == (2, 7):
        platform += "u"
    pex_request = TwoStepPexFromTargetsRequest(
        PexFromTargetsRequest(
            addresses=[field_set.address],
            internal_only=False,
            entry_point=None,
            output_filename=pex_filename,
            platforms=PexPlatforms([platform]),
            additional_args=[
                # Ensure we can resolve manylinux wheels in addition to any AMI-specific wheels.
                "--manylinux=manylinux2014",
                # When we're executing Pex on Linux, allow a local interpreter to be resolved if
                # available and matching the AMI platform.
                "--resolve-local-platforms",
            ],
        )
    )

    pex_result = await Get(TwoStepPex, TwoStepPexFromTargetsRequest, pex_request)
    input_digest = await Get(
        Digest, MergeDigests((pex_result.pex.digest, lambdex_setup.requirements_pex.digest))
    )

    # NB: Lambdex modifies its input pex in-place, so the input file is also the output file.
    result = await Get(
        ProcessResult,
        PexProcess(
            lambdex_setup.requirements_pex,
            argv=("build", "-e", field_set.handler.value, pex_filename),
            input_digest=input_digest,
            output_files=(pex_filename,),
            description=f"Setting up handler in {pex_filename}",
        ),
    )
    return CreatedAWSLambda(
        digest=result.output_digest,
        zip_file_relpath=pex_filename,
        runtime=field_set.runtime.value,
        # The AWS-facing handler function is always lambdex_handler.handler, which is the wrapper
        # injected by lambdex that manages invocation of the actual handler.
        handler="lambdex_handler.handler",
    )


@rule
async def package_python_awslambda(field_set: PythonAwsLambdaFieldSet) -> BuiltPackage:
    awslambda = await Get(CreatedAWSLambda, AWSLambdaFieldSet, field_set)
    return BuiltPackage(
        awslambda.digest,
        relpath=awslambda.zip_file_relpath,
        extra_log_info=f"  Runtime: {awslambda.runtime}\n  Handler: {awslambda.handler}",
    )


@rule(desc="Set up lambdex")
async def setup_lambdex(lambdex: Lambdex) -> LambdexSetup:
    requirements_pex = await Get(
        Pex,
        PexRequest(
            output_filename="lambdex.pex",
            internal_only=True,
            requirements=PexRequirements(lambdex.all_requirements),
            interpreter_constraints=PexInterpreterConstraints(lambdex.interpreter_constraints),
            entry_point=lambdex.entry_point,
        ),
    )
    return LambdexSetup(requirements_pex=requirements_pex)


def rules():
    return [
        *collect_rules(),
        UnionRule(AWSLambdaFieldSet, PythonAwsLambdaFieldSet),
        UnionRule(PackageFieldSet, PythonAwsLambdaFieldSet),
        *pex_from_targets.rules(),
    ]