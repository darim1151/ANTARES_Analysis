# Locked dependency workflow

`pyproject.toml` pins the direct dependency versions that pass the current
regression suite. `environment.yml` is an interactive convenience input, not a
solved Conda environment or a deployment lock. The supported Phase 1 release
path is the Linux pip/venv lock and wheelhouse described here.

Generate the release-tool, production, and test/notebook locks only in a clean
Linux x86_64 CPython 3.11.16 builder. The authoritative implementation is
`.github/workflows/linux-lock.yml`; the core application commands are:

```bash
python -m piptools compile \
  --allow-unsafe \
  --generate-hashes \
  --index-url=https://pypi.org/simple \
  --pip-args="--only-binary=:all:" \
  --resolver=backtracking \
  --strip-extras \
  --uploaded-prior-to=2026-08-20T00:00:00Z \
  --output-file=requirements/release-tools-py311-linux-x86_64.lock \
  requirements/release-tools.in

python -m piptools compile \
  --generate-hashes \
  --index-url=https://pypi.org/simple \
  --pip-args="--only-binary=:all: --no-binary=bson" \
  --resolver=backtracking \
  --strip-extras \
  --uploaded-prior-to=2026-08-20T00:00:00Z \
  --output-file=requirements/production-py311-linux-x86_64.lock \
  pyproject.toml

python -m piptools compile \
  --constraint=requirements/production-py311-linux-x86_64.lock \
  --extra=notebooks \
  --extra=test \
  --generate-hashes \
  --index-url=https://pypi.org/simple \
  --pip-args="--only-binary=:all: --no-binary=bson" \
  --resolver=backtracking \
  --strip-extras \
  --uploaded-prior-to=2026-08-20T00:00:00Z \
  --output-file=requirements/test-py311-linux-x86_64.lock \
  pyproject.toml
```

The cutoff is intentional: advancing it is a reviewed dependency-update event,
not an incidental result of rerunning CI. The workflow resolves the release
toolchain and both application locks twice and rejects byte differences. The
test lock is constrained by the production lock and is checked as an exact
superset, so tests cannot silently exercise different production transitives.
The first generation necessarily bootstraps the resolver from exact versions;
it is evidence-generation only. After the three locks are reviewed and
committed, the required final run installs the resolver from the committed
hash-checked release-tool lock before recompiling and comparing every lock.

Before accepting the locks, create fresh Python 3.11 environments on Linux,
install from the verified offline wheelhouses, build the project distributions,
run the package-origin check and complete repository integration suite, and
record:

- Linux distribution, architecture, glibc, and hosted-runner image version;
- exact Python, pip, setuptools, wheel, build, and pip-tools versions;
- Git commit and deterministic build timestamp;
- release-tool, production, test, wheelhouse, wheel, and sdist SHA-256 values;
- package inventories, `pip check`, installed CLI/import checks, and test logs.

Do not generate these files with `pip freeze` from a developer workstation and
do not deploy from `pyproject.toml` or `environment.yml` alone.

The `Generate Linux release locks` workflow performs this procedure on Ubuntu
22.04 / exact CPython 3.11.16 with GitHub Actions pinned to immutable commit
SHAs. It installs release tools from their own hash-checked lock, materializes
production and test wheelhouses from hash-verified inputs, compares all shared
wheels byte-for-byte, builds both the project wheel and sdist twice, verifies
all imports originate in a fresh installed environment, and runs the complete
repository integration suite. It uploads locks, wheelhouses, distributions,
package inventories, command logs, and a release-evidence report as one
artifact. Review and commit the three generated lock files before accepting a
release candidate.

`antares-client==1.14.0` depends on `bson==0.5.10`, for which upstream
publishes only a source distribution. The workflow permits that source input
only while keeping every other dependency binary-only. It verifies the source
archive against the lock, builds it twice with the hash-locked release
toolchain and build isolation disabled, requires the resulting production/test
wheels to be byte-identical, records their hashes, and installs only from the
offline wheelhouses. Do not broaden this exception without reviewing the new
package and recording the reason in the release evidence.

Project wheels must be byte-identical across two clean-tree builds. Setuptools
gzip timestamps are not reproducible, so project sdists are compared by safe
member path, permission mode, size, and content hash while ignoring archive
timestamps. CI then rebuilds a wheel from the sdist, requires it to equal the
direct wheel byte-for-byte, and installs that derived wheel for the smoke and
integration suites. The CPython 3.11 release builder uses
`setuptools==83.0.0`; Python 3.9 retains 80.9.0 only in its non-release,
trusted-source compatibility lane because setuptools 83 requires Python 3.10.
