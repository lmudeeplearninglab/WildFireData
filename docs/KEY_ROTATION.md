# Rotating an exposed FIRMS MAP_KEY

This project has leaked a MAP_KEY twice:

1. **Hardcoded in source** — line 292 of the original `visualize_firms_dataset.py`.
2. **Pasted into a chat/issue** — terminal output containing the `Paste MAP_KEY:`
   prompt was shared. The run transcript redacts the key; the terminal does not.

Both are rotations, but they need different follow-up. A key committed to git
needs the history purge in step 3. A key pasted somewhere else needs only step 1
— no amount of history rewriting removes it from a third party's logs.

Rotating costs about five minutes. Do it even when you think the exposure was
contained.

> Commands are given for **PowerShell** first, since this project is developed
> on Windows, with bash equivalents underneath. The PowerShell forms are not
> cosmetic translations — see the encoding warning in step 3.

---

## 0. Set the old key once

Everything below refers to the compromised key. Put it in a variable so it
never gets pasted into another file:

```powershell
$OldKey = "paste_the_compromised_key_here"
```

```bash
OLD_KEY="paste_the_compromised_key_here"
```

Do not commit this document with the literal key filled in. That is how the
first exposure happened.

## 1. Issue a new key

1. Go to <https://firms.modaps.eosdis.nasa.gov/api/map_key>.
2. Enter your email. NASA mails a new MAP_KEY, usually within a few minutes.
3. Keep the email; there is no web dashboard to look the key up again later.

FIRMS does not currently offer self-service revocation. Issuing a new key and
abandoning the old one is the available path. To have the old key positively
disabled, mail <support@earthdata.nasa.gov> with the key and a note that it was
publicly exposed — include the repo URL if it was pushed anywhere.

**If the exposure was a paste rather than a commit, you are done after this
step.** Steps 3 and 4 are still worth doing, but they cannot help with a key
that is sitting in someone else's chat log.

## 2. Put the new key in your environment, not in a file

```powershell
# Persistent, for every future session
[Environment]::SetEnvironmentVariable("FIRMS_MAP_KEY", "your_new_key", "User")

# The line above does NOT affect the window you typed it in. Either open a new
# PowerShell window, or set the current session as well:
$env:FIRMS_MAP_KEY = "your_new_key"
```

```bash
# Linux / macOS — add to ~/.bashrc or ~/.zshrc
export FIRMS_MAP_KEY=your_new_key_here
```

Verify:

```powershell
$env:FIRMS_MAP_KEY
python visualize_firms_dataset_v3.py --area eaton --points --days 2
```

```bash
echo $FIRMS_MAP_KEY
python visualize_firms_dataset_v3.py --area eaton --points --days 2
```

Startup prints your transaction count against the quota, which confirms the key
is live.

For a project-local setup, a `.env` file works as long as it is gitignored:

```powershell
Set-Content -Path ".env" -Value "FIRMS_MAP_KEY=your_new_key" -Encoding utf8
Add-Content -Path ".gitignore" -Value ".env"
```

```bash
echo "FIRMS_MAP_KEY=your_new_key" > .env
echo ".env" >> .gitignore
```

## 3. Purge the old key from git history

**Rotating alone is not enough if the key was ever committed.** It stays in
history and in every clone and fork. Check first:

```powershell
git log -S $OldKey --oneline --all
```

```bash
git log -S "$OLD_KEY" --oneline --all
```

No output means it was never committed — skip to step 4.

### The PowerShell encoding trap

`echo "text" > file.txt` in PowerShell writes **UTF-16LE with a byte-order
mark**, not plain text. `git filter-repo` reads that file as garbage: it will
either match nothing and report success, or fail with an unhelpful parse error.
Either way you end up believing the key was purged when it was not.

Always write the rules file with an explicit encoding:

```powershell
# Back up first. History rewriting is destructive and irreversible.
Copy-Item -Recurse WildFirePred WildFirePred-backup

pip install git-filter-repo

Set-Content -Path "$env:TEMP\replace.txt" `
            -Value "$OldKey==>REDACTED_ROTATED_KEY" `
            -Encoding ascii

git filter-repo --replace-text "$env:TEMP\replace.txt"

# filter-repo drops remotes deliberately; re-add and force-push
git remote add origin https://github.com/you/your-repo.git
git push --force --all
git push --force --tags

Remove-Item "$env:TEMP\replace.txt"
```

On PowerShell 7+, `-Encoding utf8NoBOM` also works. On Windows PowerShell 5.1,
use `ascii` — its `utf8` writes a BOM.

```bash
cp -r your-repo your-repo-backup
pip install git-filter-repo

cd your-repo
echo "$OLD_KEY==>REDACTED_ROTATED_KEY" > /tmp/replace.txt
git filter-repo --replace-text /tmp/replace.txt

git remote add origin git@github.com:you/your-repo.git
git push --force --all
git push --force --tags
rm /tmp/replace.txt
```

`git filter-repo` refuses to run on a repo with uncommitted changes, so commit
or stash first.

Then tell every collaborator to re-clone. Old clones still contain the key, and
pulling will not fix that — merging a rewritten history reintroduces the old
commits.

If the repo was ever public on GitHub, rewriting history still does not purge
GitHub's cached views of orphaned commits. Open a support ticket asking them to
garbage-collect, or accept that the key is public and rely on the rotation.
Either way the new key is safe — that is the point of rotating.

## 4. Stop it happening again

Add a pre-commit hook that blocks secrets. In PowerShell, heredocs become
here-strings: the closing `'@` must sit at column zero with no leading
whitespace, or the string does not terminate.

```powershell
pip install detect-secrets pre-commit

$config = @'
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
'@
Set-Content -Path ".pre-commit-config.yaml" -Value $config -Encoding ascii

# Do NOT pipe the scan through Set-Content or `>`. Let a non-PowerShell
# writer produce the file -- see the warning below.
cmd /c "detect-secrets scan > .secrets.baseline"

pre-commit install
```

### The baseline must not get a BOM

`.secrets.baseline` is JSON that `detect-secrets` parses on every commit. On
**Windows PowerShell 5.1**, both `>` and `Set-Content -Encoding utf8` prepend a
byte-order mark, and `json.loads` then fails on the very first character:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
detect_secrets.exceptions.UnableToReadBaselineError
```

`-Encoding utf8` is BOM-less on PowerShell 7+ but **not** on 5.1, so it is not
safe to rely on. Check with `$PSVersionTable.PSVersion`. The dependable options,
in order of preference:

```powershell
# 1. Let cmd write it -- raw bytes, no BOM, works on every PowerShell
cmd /c "detect-secrets scan > .secrets.baseline"

# 2. Or have Python write it
python -c "import subprocess; open('.secrets.baseline','w',encoding='utf-8',newline='\n').write(subprocess.run(['detect-secrets','scan'],capture_output=True,text=True).stdout)"

# 3. PowerShell 7+ only
detect-secrets scan | Out-File .secrets.baseline -Encoding utf8NoBOM
```

Verify it parses before going any further:

```powershell
python -c "import json; d=json.load(open('.secrets.baseline')); print(len(d.get('results',{})), 'files with findings')"
```

To inspect a suspect file: `Format-Hex .secrets.baseline -Count 8`. Leading
`EF BB BF` is a UTF-8 BOM; `FF FE` is UTF-16.

```bash
pip install detect-secrets pre-commit

cat > .pre-commit-config.yaml <<'YAML'
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
YAML

detect-secrets scan > .secrets.baseline
pre-commit install
```

If this lives on GitHub, also switch on **Settings → Code security → Secret
scanning** and **Push protection**, which rejects pushes containing recognized
credential patterns.

## 5. Confirm the code no longer contains it

The test suite enforces this:

```powershell
pytest -q test_firms.py -k credentials
```

`test_no_hardcoded_credentials` fails if the old key reappears or if any
32-hex-character literal is assigned to a MAP_KEY variable. Keep it in CI so a
future copy-paste cannot reintroduce the pattern.

## 6. Check what else you are about to share

The second exposure came from pasting terminal output, so it is worth knowing
what is and is not safe to share.

**Safe.** `report.txt` and `build_report.txt` mask anything matching a
credential before it reaches the file: a 32-hex-character string, a `MAP_KEY=`
query parameter, an `api_key:` assignment. Answers to prompts mentioning key,
token, secret, password or credential are recorded as `<redacted>` rather than
written out. So the transcript is the thing to send when asking for help.

**Not safe.** Raw terminal output. Your terminal echoes what you type, and the
redaction only applies to the file. If you copy from the console rather than
from the transcript, the key comes with it — which is exactly what happened.

**Never share.** `~/.config/earthengine/credentials` (or
`C:\Users\<you>\.config\earthengine\credentials`). That is a live OAuth token
for your Google account, not just Earth Engine. Add these to `.gitignore`:

```
.env
report.txt
build_report.txt
**/credentials
```

Transcripts are gitignored above out of caution: they are redacted for
credentials, but they still record local paths, fire names and date ranges that
you may not want in a public repo.

**Do commit** `.pre-commit-config.yaml` and `.secrets.baseline`. The baseline
stores hashes of findings you have already reviewed, not the secrets
themselves; gitignoring it means every contributor re-triages the same
findings on every commit, which is how people end up disabling the hook.
