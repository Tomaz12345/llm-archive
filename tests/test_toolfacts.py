"""The extractor: paths, commands, and the per-tool rules.

This is where the feature will drift as tools are added and renamed, so most of these
tests are named after the wrong answer they prevent rather than the code they cover.
"""

from llm_archive.core.toolinput import (
    CommandFact, FileFact, extract, is_known, normalise_path, split_argv0,
)

WS_WIN = "c:/users/tomaz/documents/projekti/llm_sessions_grouping"
WS_SSH = "ssh-remote+vicos-proxy-30055/home/tomazpoljansek/proj"


# --------------------------------------------------------------- normalise_path

def test_a_windows_path_is_casefolded_so_two_drive_cases_are_one_file():
    """The archive really holds the same file under both `C:` and `c:`;
    keying on the raw string splits every Windows project in two."""
    upper = normalise_path(r"C:\Users\tomaz\Documents\x.py")
    lower = normalise_path(r"c:/users/tomaz/documents/x.py")
    assert upper.norm == lower.norm == "c:/users/tomaz/documents/x.py"
    assert upper.base == "x.py"
    assert upper.path == r"C:\Users\tomaz\Documents\x.py"   # the literal is preserved


def test_the_drive_letter_is_kept_because_c_and_d_are_different_disks():
    assert normalise_path("C:/x/a.py").norm != normalise_path("D:/x/a.py").norm


def test_a_vscode_remote_uri_becomes_a_host_qualified_key():
    """This is the shape `workspace.key` already uses for remote roots, which is the
    only reason `rel` can be a prefix strip."""
    got = normalise_path(
        "vscode-remote://ssh-remote%2Bvicos-proxy-30055/home/tomazpoljansek/proj/train.py")
    assert got.host_key == "ssh-remote+vicos-proxy-30055"
    assert got.norm == "ssh-remote+vicos-proxy-30055/home/tomazpoljansek/proj/train.py"


def test_a_vscode_uri_object_is_accepted_directly():
    """VS Code records `{path, scheme, authority}` and nothing else; re-serialising it
    to a URI just to parse it back would be the long way round."""
    got = normalise_path({"path": "/home/tomazpoljansek/proj/train.py",
                          "scheme": "vscode-remote",
                          "authority": "ssh-remote+vicos-proxy-30055"})
    assert got.norm == "ssh-remote+vicos-proxy-30055/home/tomazpoljansek/proj/train.py"


def test_a_windows_path_and_its_wsl_twin_are_not_the_same_file():
    win = normalise_path(r"C:\home\tomaz\x.py")
    wsl = normalise_path(r"\\wsl$\Ubuntu\home\tomaz\x.py")
    assert wsl.host_key == "wsl+ubuntu"
    assert wsl.norm == "wsl+ubuntu/home/tomaz/x.py"
    assert win.norm != wsl.norm


def test_a_plain_file_uri_has_no_host():
    got = normalise_path("file:///home/tomaz/args_file.txt.example")
    assert got.host_key is None
    assert got.norm == "/home/tomaz/args_file.txt.example"


def test_dot_segments_are_resolved_without_touching_the_disk():
    assert normalise_path("/a/b/../c/./d.py").norm == "/a/c/d.py"


def test_a_trailing_slash_does_not_make_a_second_directory():
    assert normalise_path("/a/b/").norm == normalise_path("/a/b").norm


def test_a_relative_path_is_joined_to_the_cwd_when_one_is_known():
    got = normalise_path("src/main.py", cwd="C:/proj")
    assert got.norm == "c:/proj/src/main.py"


def test_a_relative_path_survives_with_no_cwd():
    """Codex patches record repo-relative paths and no cwd at all; dropping them would
    lose every file in a 12-file patch."""
    got = normalise_path("app/src/Main.kt")
    assert got.norm == "app/src/main.kt"
    assert got.base == "main.kt"


def test_the_same_relative_file_from_two_roots_shares_a_rel():
    """The cross-machine payoff: `code/train.py` edited on the SSH box and on Windows
    must land in one row of the hot-files list, not two."""
    ssh = normalise_path({"path": "/home/tomazpoljansek/proj/code/train.py",
                          "scheme": "vscode-remote",
                          "authority": "ssh-remote+vicos-proxy-30055"},
                         workspace_key=WS_SSH)
    win = normalise_path(r"C:\Users\tomaz\Documents\Projekti\LLM_sessions_grouping\code\train.py",
                         workspace_key=WS_WIN)
    assert ssh.rel == win.rel == "code/train.py"
    assert ssh.norm != win.norm


def test_a_path_outside_the_workspace_has_no_rel():
    assert normalise_path("C:/elsewhere/x.py", workspace_key=WS_WIN).rel is None


def test_normalise_path_normalises_rather_than_detects():
    """A bare directory name is a real relative path — `Grep(path="llm_archive")` is a
    real search. Deciding whether a string IS a path belongs to the rule table, which
    knows which key it read; this function only has the string."""
    assert normalise_path("llm_archive").norm == "llm_archive"
    assert normalise_path("./llm_archive").norm == "llm_archive"
    assert normalise_path("") is None
    assert normalise_path(None) is None
    assert normalise_path(12) is None


def test_an_unlisted_tools_bare_word_is_not_promoted_to_a_file():
    """The fallback has nothing telling it the value is a path, so a bare word — far
    more often a mode or an id — must not become a touched file."""
    assert extract("claude_code", "SomeNewTool", {"path": "auto"}).files == ()
    assert extract("claude_code", "SomeNewTool", {"path": "src/x.py"}).files         == (FileFact("src/x.py", "other"),)


def test_a_tool_that_touches_nothing_is_not_reported_as_drift():
    """702 calls of WebFetch/WebSearch/AskUserQuestion would drown the one signal that
    matters: a real file tool the rule table has never seen."""
    assert is_known("claude_code", "WebFetch")
    assert not extract("claude_code", "WebFetch", {"url": "https://x/y.py"})


def test_a_path_quoted_inside_prose_is_not_a_touched_file():
    """`Monitor(until="The file C:/…/x.mp4 exists…")` is the shell-text trap wearing a
    different hat."""
    assert not extract("claude_code", "Monitor",
                       {"until": "The file C:/x/skijump.mp4 exists and is unlocked."})


def test_send_user_file_records_every_file_it_sent():
    facts = extract("claude_code", "SendUserFile",
                    {"files": ["C:/tmp/a.png", "C:/tmp/b.png"], "caption": "x"})
    assert facts.files == (FileFact("C:/tmp/a.png", "read"),
                           FileFact("C:/tmp/b.png", "read"))


def test_an_http_url_is_not_a_touched_file():
    assert normalise_path("https://example.com/a.py") is None


# ------------------------------------------------------------------ split_argv0

def test_argv0_ignores_env_prefixes_and_sudo():
    assert split_argv0("sudo FOO=1 git commit -m x") == ("git", "commit")


def test_argv0_strips_a_path_and_an_exe_suffix():
    """A Windows command line must survive POSIX shlex, which reads its separators as
    escapes and would otherwise report `c:pythonpython`. `-m pytest` is kept as the
    subcommand because bare `python` says nothing about what ran."""
    assert split_argv0(r"C:\Python\python.exe -m pytest") == ("python", "pytest")


def test_argv0_stops_at_the_first_pipeline_stage():
    """`ls | grep .py` is an `ls`, not a `grep`."""
    assert split_argv0("ls -la | grep .py")[0] == "ls"
    assert split_argv0("pytest -q | tee out.txt")[0] == "pytest"


def test_argv0_survives_an_unbalanced_quote():
    """Real command lines contain stray quotes; shlex raises ValueError on them and a
    crash here would take down the whole index."""
    assert split_argv0('git commit -m "unclosed') == ("git", "commit")


def test_argv0_of_a_powershell_pipeline():
    assert split_argv0("Get-ChildItem -Force", shell="powershell") == ("get-childitem", None)


def test_a_subcommand_is_never_a_flag_or_a_path():
    assert split_argv0("pytest -q tests/") == ("pytest", None)
    assert split_argv0("docker run -it ubuntu") == ("docker", "run")


def test_an_empty_command_is_not_a_crash():
    assert split_argv0("") == ("", None)
    assert split_argv0("   ") == ("", None)


# ------------------------------------------------------------------ rule table

def test_claude_code_file_tools_map_to_their_verbs():
    assert extract("claude_code", "Read", {"file_path": "/a/b.py"}).files \
        == (FileFact("/a/b.py", "read"),)
    assert extract("claude_code", "Write", {"file_path": "/a/b.py"}).files \
        == (FileFact("/a/b.py", "write"),)
    assert extract("claude_code", "Edit", {"file_path": "/a/b.py"}).files \
        == (FileFact("/a/b.py", "edit"),)


def test_a_bash_call_yields_the_whole_command_not_a_truncated_one():
    """The 300-character summary is exactly what this feature exists to get past."""
    long = "python -c " + ("x = 1; " * 200)
    facts = extract("claude_code", "Bash", {"command": long})
    assert facts.commands[0].text == long.strip()
    assert len(facts.commands[0].text) > 300


def test_a_shell_command_yields_no_touched_files():
    """Pins the v1 decision so relaxing it is a choice, not a regression.
    A commit message that names a file did not touch that file."""
    facts = extract("claude_code", "Bash",
                    {"command": 'git commit -m "fix src/foo.py" && rm old.txt'})
    assert facts.files == ()
    assert len(facts.commands) == 1


def test_a_claude_web_artifact_is_not_a_shell_command():
    """`artifacts` takes an input.command of create/update/rewrite. A rule keyed on the
    payload's keys rather than on the tool invents 64 shell commands."""
    facts = extract("claude_web", "artifacts", {"command": "create", "id": "x"})
    assert facts.commands == ()
    assert facts.files == ()


def test_an_mcp_tools_arguments_are_left_alone():
    facts = extract("claude_code", "mcp__blender__execute_blender_code", {"code": "x"})
    assert not facts


def test_codex_apply_patch_yields_one_row_per_file():
    patch = (
        "*** Begin Patch\n"
        "*** Update File: app/src/Main.kt\n"
        "@@\n-a\n+b\n"
        "*** Add File: app/src/New.kt\n"
        "+hello\n"
        "*** Delete File: app/src/Old.kt\n"
        "*** End Patch\n")
    facts = extract("codex", "apply_patch", patch)
    assert facts.files == (
        FileFact("app/src/Main.kt", "edit"),
        FileFact("app/src/New.kt", "write"),
        FileFact("app/src/Old.kt", "delete"))


def test_codex_records_a_command_as_argv():
    facts = extract("codex", "shell_command",
                    {"command": ["bash", "-lc", "pytest -q"], "workdir": "/proj"})
    assert facts.commands[0].text == "bash -lc pytest -q"
    assert facts.commands[0].cwd == "/proj"


def test_a_json_string_payload_is_parsed():
    """Codex hands `arguments` over as a JSON string, not a dict."""
    facts = extract("opencode", "read", '{"filePath": "/a/b.py"}')
    assert facts.files == (FileFact("/a/b.py", "read"),)


def test_vscode_reads_its_paths_out_of_the_uris_map():
    payload = {"phase": "complete",
               "uris": [{"path": "/home/t/proj/train.py", "scheme": "vscode-remote",
                         "authority": "ssh-remote+jon"}]}
    facts = extract("vscode_chat", "copilot_readFile", payload)
    assert len(facts.files) == 1 and facts.files[0].action == "read"


def test_a_prepare_block_does_not_double_count_the_call():
    """VS Code writes prepareToolInvocation AND toolInvocationSerialized for the same
    call. Counting both doubles every file touch the panel ever made."""
    payload = {"phase": "prepare", "uris": [{"path": "/a/b.py"}]}
    assert extract("vscode_chat", "copilot_readFile", payload).files == ()


def test_apply_patch_reads_the_path_backfilled_from_the_edit_group():
    """copilot_applyPatch carries no path of its own - 0 of 354 blocks have `uris`."""
    payload = {"phase": "complete",
               "editedUris": [{"path": "/home/t/proj/train.py"}]}
    facts = extract("vscode_chat", "copilot_applyPatch", payload)
    assert len(facts.files) == 1
    assert facts.files[0].action == "edit"


def test_run_in_terminal_reads_the_command_and_the_exit_code():
    """`isComplete` is true for a command that exited 127, so it cannot be `ok`."""
    payload = {"phase": "complete", "terminal": {
        "commandLine": {"original": "mypy src"}, "language": "bash",
        "terminalCommandState": {"exitCode": 127}}}
    facts = extract("vscode_chat", "run_in_terminal", payload)
    assert facts.commands == (CommandFact("mypy src", "posix", None, False),)


def test_a_glob_search_is_not_a_touched_file():
    """`**/src/**` is not a file, and the table's name is a promise."""
    assert not extract("vscode_chat", "copilot_findTextInFiles",
                       {"phase": "complete", "query": "x", "includePattern": "**/src/**"})
    assert not extract("claude_code", "Grep", {"pattern": "def load"})


def test_a_grep_with_a_real_directory_is_recorded_as_a_search():
    facts = extract("claude_code", "Grep", {"pattern": "def load", "path": "src/core"})
    assert facts.files == (FileFact("src/core", "search"),)


def test_a_github_file_is_qualified_by_its_repo():
    facts = extract("copilot_web", "getfile",
                    {"repo": "tomaz12345/pacman_agent", "path": "README.md"})
    got = normalise_path(facts.files[0].path,
                         workspace_key="github.com/tomaz12345/pacman_agent")
    assert got.host_key == "github.com"
    assert got.rel == "readme.md"


def test_an_unknown_tool_that_names_a_file_still_counts_as_drift():
    facts = extract("claude_code", "SomeNewTool", {"file_path": "/a/b.py"})
    assert facts.files == (FileFact("/a/b.py", "other"),)
    assert not is_known("claude_code", "SomeNewTool")
    assert is_known("claude_code", "Read")


def test_a_malformed_payload_is_never_an_exception():
    for bad in (None, 12, [], "not json", b"\xff\xfe", {"file_path": 5}):
        assert extract("claude_code", "Read", bad) is not None


def test_a_cd_prefix_is_not_the_command():
    """Nearly every Bash call in this archive opens `cd <project> && ...`, which made
    `cd` 4,444 of 6,695 derived commands -- a ranking that answered "what do I run"
    with "you change directory a lot"."""
    assert split_argv0("cd /x && pytest -q") == ("pytest", None)
    assert split_argv0("cd a && cd b && npm run build") == ("npm", "run")
    assert split_argv0("export FOO=1 && make") == ("make", None)
    assert split_argv0("cd /x; ./run.sh")[0] == "run"


def test_a_bare_cd_is_still_a_cd():
    """Skipping the prefix must not eat the whole command when there is no next stage."""
    assert split_argv0("cd /x") == ("cd", None)
    assert split_argv0("cd") == ("cd", None)


def test_a_pipe_is_not_a_sequence():
    """In `ls | grep x` the program is `ls`; only && ; and newline start a new command."""
    assert split_argv0("ls -la | grep .py")[0] == "ls"
    assert split_argv0("cd /x && ls | grep .py")[0] == "ls"


def test_a_variable_is_not_a_subcommand():
    """`python "$f"` runs python; `$f` is a path standing in a variable."""
    assert split_argv0('f="/p/x"; python "$f"') == ("python", None)


def test_an_assignment_only_stage_is_a_prefix_not_a_command():
    """476 commands were dropped entirely because their opening stage was nothing but
    a variable assignment, which strips to no tokens at all."""
    assert split_argv0('f="/p/x.txt"; wc -l "$f"')[0] == "wc"
    assert split_argv0("FOO=1 BAR=2 make -j4") == ("make", None)


def test_a_shell_loop_reports_the_program_inside_it():
    """`for f in *.py; do python "$f"; done` is a python run. Naively taking the first
    stage says `for`; skipping whole keyword stages says `done`."""
    assert split_argv0('for f in *.py; do python "$f"; done')[0] == "python"
    assert split_argv0("while read l; do echo $l; done")[0] == "echo"
    assert split_argv0("if [ -f x ]; then cat x; fi")[0] == "cat"
