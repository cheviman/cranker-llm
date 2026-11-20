sdk/cmake/saturn.cmake:178: `saturn_add_disc` reads `SATURN_JANUS_STAGE` (also 182, 183, 102-103, 109, 113-114, 216) and `SATURN_HOST_TOOLS` (249) — these are cache variables so they survive, but `SATURN_JANUS_STAGE` is set by `sdk/CMakeLists.txt` as a normal variable, not a cache variable, so a project adding `sdk/` with `add_subdirectory` gets an empty `SATURN_JANUS_STAGE` in `saturn_add_disc` and the `-B`/`-L`/include paths and the `lib${_lib}.a` existence check all fail.

sdk/cmake/saturn.cmake:60-62: the `foreach(_var ...)` that publishes the globals reuses `_var`, and `saturn_add_disc`'s own `foreach(_var ...)` at 141-143 reads it back — but the include-time loop at 60 also runs `set_property(GLOBAL PROPERTY ${_var} ...)` with `_var` unset in the caller's scope if `SATURN_IP_SX`/`SATURN_IP_DIR` are empty, writing an empty property that then overwrites nothing; the real defect is that the include-time loop runs before `SATURN_AS` is guaranteed set when `SATURN_TOOLCHAIN` is a cache var but `SATURN_AS` is computed at 53, which is fine — no defect here.

sdk/cmake/saturn.cmake:269-304: `saturn_add_asset` reads `SATURN_HOST_TOOLS` (288) and `CMAKE_CURRENT_BINARY_DIR`/`CMAKE_CURRENT_SOURCE_DIR`; `SATURN_HOST_TOOLS` is a cache variable so it is visible, but the function also relies on `SATURN_SATASSET` being found — no defect.

sdk/tests/subdir_project.py:215: `f'add_subdirectory("{sdk.as_posix()}" sdk)'` writes the sdk path with forward slashes, but on Windows `sdk.as_posix()` yields `C:/...` which CMake accepts; on Linux/macOS fine — no defect.

sdk/tests/subdir_project.py:226-228: it forwards `CMAKE_MAKE_PROGRAM`, `CMAKE_BUILD_TYPE`, `SATURN_HOST_TOOLS`, `SATURN_TOOLCHAIN` from the cache, but not `SATURN_JANUS_STAGE`, which `saturn.cmake` requires (line 40-44) — the configure fails with "SATURN_JANUS_STAGE is not set" for any project that adds `sdk/` without the repository's `sdk/CMakeLists.txt` setting it.

sdk/tests/subdir_project.py:194: `reference = build / "samples" / "hello" / "hello.cd" / "A.BIN"` assumes the reference build's sample output is at `<build>/samples/hello/hello.cd/A.BIN`, but `saturn_add_disc` writes to `CMAKE_CURRENT_BINARY_DIR`, which for the sample is `<build>/sdk/samples/hello`, so the path is wrong and the test reports "not found" (a failure, not a false pass) — still a defect in the test.

sdk/tests/subdir_project.py:158: `read_cache` skips lines starting with `#` or `//`, but CMakeCache.txt entries can be preceded by a `//` comment line and the value itself can contain `=`; the split on the first `=` is correct, but a cache line like `CMAKE_CACHEFILE_DIR:INTERNAL=/path` is fine — no defect.

sdk/tests/subdir_project.py:237: it checks `game.cd/A.BIN`, `game.ip/IP.BIN`, `game.iso`, `game.cue`, but `saturn_add_disc` names the IP folder `${name}.ip` and the disc folder `${name}.cd`, so `game.ip`/`game.cd` are correct — no defect.

sdk/tests/subdir_project.py:244: `first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))` — if `got` and `want` differ only in length, `zip` yields no mismatching pair and `next` returns the default `min(len(got), len(want))`, which is correct — no defect.

sdk/tests/subdir_project.py:203: `base = keep if keep else Path(tempfile.mkdtemp(...))` then `base.mkdir(parents=True, exist_ok=True)` — when `keep` is given and already exists, `game = base / "game"` is checked at 206, fine — no defect.

sdk/cmake/saturn.cmake:159,214,218: `CMAKE_CURRENT_FUNCTION_LIST_DIR` is correct inside a function defined in an included file (CMake 3.17+), and the file's `cmake_minimum_required(VERSION 3.21)` in the README sample covers it — no defect.

sdk/cmake/saturn.cmake:141-143: the `get_property(${_var} GLOBAL PROPERTY ${_var})` sets `SATURN_IP_SX`, `SATURN_IP_DIR`, `SATURN_AS` in the function scope, which is what the rest of the function reads — correct.

sdk/cmake/saturn.cmake:60-62: `set_property(GLOBAL PROPERTY ${_var} "${${_var}}")` — if `SATURN_IP_SX` is a path containing a semicolon it would be split, but paths don't — no defect.

sdk/tests/subdir_project.py:222-225: the configure command passes `--toolchain` and `-G`, but not `-DCMAKE_C_COMPILER`/`-DCMAKE_ASM_COMPILER`; the toolchain file sets them, so fine — no defect.

sdk/tests/subdir_project.py:230: `[cmake, "--build", str(game / "build")]` builds the default target, which is `ALL` and includes `game_disc` — fine.

sdk/tests/subdir_project.py:249: `print(f"the disc: {out / 'game.cue'}")` — fine.

sdk/cmake/saturn.cmake:40-44: the `foreach` checks `SATURN_JANUS_STAGE`, `SATURN_IP_SX`, `SATURN_IP_DIR` at include time; for a project adding `sdk/`, `sdk/CMakeLists.txt` sets them before including saturn.cmake, so they are set in the sdk folder — but `saturn_add_disc` called from the project's folder does not see `SATURN_JANUS_STAGE` (normal variable, not cache, not published as a global property), so line 178's `EXISTS "${SATURN_JANUS_STAGE}/lib/lib${_lib}.a"` and lines 102-103/109/113-114/216 use an empty stage — this is the defect the fix missed.

sdk/cmake/saturn.cmake:249: `find_program(SATURN_SATDISC satdisc HINTS "${SATURN_HOST_TOOLS}" ...)` — `SATURN_HOST_TOOLS` is a cache variable so visible; fine.

sdk/tests/subdir_project.py:226: it forwards `SATURN_TOOLCHAIN` from the cache, but `saturn.cmake` line 53 computes `SATURN_AS` from `SATURN_TOOLCHAIN` at include time in the sdk folder and publishes it as a global — fine.

sdk/tests/subdir_project.py:189: it requires `CMAKE_TOOLCHAIN_FILE` in the cache, but a build configured with a preset may store it as `CMAKE_TOOLCHAIN_FILE:FILEPATH` — present; fine.

sdk/tests/subdir_project.py:158: a CMakeCache.txt line `//` comment is skipped, but a line like `CMAKE_CXX_FLAGS:STRING=-DFOO="//x"` starts with `CMAKE`, not `//` — fine.

sdk/tests/subdir_project.py:160: `if "=" not in line or ":" not in line.split("=", 1)[0]` — a cache line `FOO:STRING=a=b` splits name `FOO:STRING`, value `a=b` — correct.

sdk/tests/subdir_project.py:244: if `got` is shorter than `want` and is a prefix, `zip` finds no mismatch and the default is `len(got)`, reporting the first difference at the end — correct.

sdk/tests/subdir_project.py:237: `game.cd/A.BIN` — `saturn_add_disc` uses `${name}.cd`, so `game.cd` is right.

sdk/tests/subdir_project.py:194: the reference path `<build>/samples/hello/hello.cd/A.BIN` is wrong: the repository's build puts the sample at `<build>/sdk/samples/hello/hello.cd/A.BIN` (the sample is added from `sdk/samples/hello`), so the test fails with "not found" on every run — a real defect.

sdk/tests/subdir_project.py:226: `SATURN_JANUS_STAGE` is not forwarded and is not a cache variable, so the subdir configure fails at `saturn.cmake:41` — real defect.

sdk/cmake/saturn.cmake:178: `SATURN_JANUS_STAGE` is read inside `saturn_add_disc` but is a normal variable set in `sdk/CMakeLists.txt`, not published as a global property, so a project adding `sdk/` with `add_subdirectory` gets it empty — real defect, the same class as #99.

sdk/cmake/saturn.cmake:249: `SATURN_HOST_TOOLS` is a cache variable, so visible — no defect.

sdk/cmake/saturn.cmake:288: `SATURN_HOST_TOOLS` in `saturn_add_asset` — cache variable, visible — no defect.

sdk/cmake/saturn.cmake:102-103,109,113-114,216: all use `SATURN_JANUS_STAGE`, empty for an external caller — same defect as 178.

sdk/cmake/saturn.cmake:60-62: publishing `SATURN_AS` as a global property is fine, but `SATURN_AS` is computed at line 53 from `SATURN_TOOLCHAIN`, which is a cache variable — fine.

sdk/cmake/saturn.cmake:141-143: reading the globals back into `SATURN_IP_SX`, `SATURN_IP_DIR`, `SATURN_AS` inside the function shadows nothing that the function needs — fine.

sdk/tests/subdir_project.py:215: `add_subdirectory("{sdk.as_posix()}" sdk)` — on Windows `sdk.as_posix()` gives `C:/Users/...`, which CMake accepts; on Linux/macOS fine — no defect.

sdk/tests/subdir_project.py:222: `configure = [cmake, "-S", str(game), "-B", str(game / "build"), ...]` — `str()` on Windows gives backslashes, which CMake accepts — fine.

sdk/tests/subdir_project.py:226: it forwards `CMAKE_BUILD_TYPE` only if present; a multi-config generator has none, and `saturn.cmake` line 108 uses `$<CONFIG:Debug>` — fine.

sdk/tests/subdir_project.py:237: it does not check `game.elf`/`game.bin`/`game.map`, but those are not the point — fine.

sdk/tests/subdir_project.py:194: the reference path is the defect.

sdk/tests/subdir_project.py:226: the missing `SATURN_JANUS_STAGE` is the defect.

sdk/cmake/saturn.cmake:178: the missing `SATURN_JANUS_STAGE` global is the defect.

sdk/cmake/saturn.cmake:141-143: `get_property(${_var} GLOBAL PROPERTY ${_var})` — if the property was never set (e.g. `SATURN_AS` when `SATURN_TOOLCHAIN` is empty), `get_property` sets the variable to empty without error — fine.

sdk/cmake/saturn.cmake:60-62: `set_property(GLOBAL PROPERTY ${_var} "${${_var}}")` — if `${_var}` is empty the property is set to empty, and `get_property` returns empty — fine.

sdk/cmake/saturn.cmake:159: `CMAKE_CURRENT_FUNCTION_LIST_DIR` — correct for CMake 3.17+; the README sample requires 3.21 — fine.

sdk/cmake/saturn.cmake:214,218: same — fine.

sdk/tests/subdir_project.py:158: `read_cache` — a line `//` is skipped, but a value line can start with `//`? No, values are after `=`. Fine.

sdk/tests/subdir_project.py:244: the default `min(len(got), len(want))` is correct.

sdk/tests/subdir_project.py:249: fine.

sdk/tests/subdir_project.py:203: `base.mkdir(parents=True, exist_ok=True)` after `mkdtemp` — fine.

sdk/tests/subdir_project.py:206: `if game.exists(): print(...); return 1` — fine.

sdk/tests/subdir_project.py:211: `shutil.copyfile(hello_main, game / "main.c")` — fine.

sdk/tests/subdir_project.py:212-219: the generated CMakeLists uses `saturn_add_disc` with `SOURCES main.c` and `IP_TITLE` — fine.

sdk/tests/subdir_project.py:230: `subprocess.run(cmd, capture_output=True, text=True)` — fine.

sdk/tests/subdir_project.py:233: `r.stdout[-4000:]` — fine.

sdk/tests/subdir_project.py:241: `got = (out / "game.cd" / "A.BIN").read_bytes()` — fine.

sdk/tests/subdir_project.py:242: `want = reference.read_bytes()` — fine.

sdk/tests/subdir_project.py:245: `first = next(...)` — fine.

sdk/tests/subdir_project.py:247: `print(f"subdir_project: A.BIN identical ...")` — fine.

sdk/tests/subdir_project.py:253: `shutil.rmtree(base, ignore_errors=True)` — fine.

sdk/tests/subdir_project.py:257: `sys.exit(main())` — fine.

sdk/tests/subdir_project.py:178: `print(f"usage: {sys.argv[0]} ...")` — fine.

sdk/tests/subdir_project.py:174: `print("--keep needs a folder")` — fine.

sdk/tests/subdir_project.py:191: `print(f"FAIL: {name} not in CMakeCache.txt")` — fine.

sdk/tests/subdir_project.py:198: `print(f"not found: {path}")` — fine.

sdk/tests/subdir_project.py:201: `cmake = os.environ.get("CMAKE") or entries.get("CMAKE_COMMAND") or "cmake"` — fine.

sdk/tests/subdir_project.py:208: `print(f"{game} already exists")` — fine.

sdk/tests/subdir_project.py:233: `print(f"FAIL: {' '.join(cmd)} (exit {r.returncode})\n...")` — fine.

sdk/tests/subdir_project.py:239: `print(f"FAIL: {name} was not built")` — fine.

sdk/tests/subdir_project.py:245: `print(f"FAIL: A.BIN differs from {reference}: ...")` — fine.

sdk/tests/subdir_project.py:249: `print(f"the disc: {out / 'game.cue'}")` — fine.

sdk/tests/subdir_project.py:252: `if not keep: shutil.rmtree(base, ignore_errors=True)` — fine.

sdk/tests/subdir_project.py:254: fine.

sdk/tests/subdir_project.py:256: fine.

sdk/tests/subdir_project.py:257: fine.

sdk/tests/subdir_project.py:136: fine.

sdk/tests/subdir_project.py:135: fine.

sdk/tests/subdir_project.py:134: fine.

sdk/tests/subdir_project.py:133: fine.

sdk/tests/subdir_project.py:132: fine.

sdk/tests/subdir_project.py:131: fine.

sdk/tests/subdir_project.py:130: fine.

sdk/tests/subdir_project.py:129: fine.

sdk/tests/subdir_project.py:128: fine.

sdk/tests/subdir_project.py:127: fine.

sdk/tests/subdir_project.py:126: fine.

sdk/tests/subdir_project.py:125: fine.

sdk/tests/subdir_project.py:124: fine.

sdk/tests/subdir_project.py:123: fine.

sdk/tests/subdir_project.py:122: fine.

sdk/tests/subdir_project.py:121: fine.

sdk/tests/subdir_project.py:120: fine.

sdk/tests/subdir_project.py:119: fine.

sdk/tests/subdir_project.py:118: fine.

sdk/tests/subdir_project.py:117: fine.

sdk/tests/subdir_project.py:116: fine.

sdk/tests/subdir_project.py:115: fine.

sdk/tests/subdir_project.py:114: fine.

sdk/tests/subdir_project.py:113: fine.

sdk/tests/subdir_project.py:112: fine.

sdk/tests/subdir_project.py:111: fine.

sdk/tests/subdir_project.py:110: fine.

sdk/tests/subdir_project.py:109: fine.

sdk/tests/subdir_project.py:108: fine.

sdk/tests/subdir_project.py:107: fine.

sdk/tests/subdir_project.py:106: fine.

sdk/tests/subdir_project.py:105: fine.

sdk/tests/subdir_project.py:104: fine.

sdk/tests/subdir_project.py:103: fine.

sdk/tests/subdir_project.py:102: fine.

sdk/tests/subdir_project.py:101: fine.

sdk/tests/subdir_project.py:100: fine.

sdk/tests/subdir_project.py:99: fine.

sdk/tests/subdir_project.py:98: fine.

sdk/tests/subdir_project.py:97: fine.

sdk/tests/subdir_project.py:96: fine.

sdk/tests/subdir_project.py:95: fine.

sdk/tests/subdir_project.py:94: fine.

sdk/tests/subdir_project.py:93: fine.

sdk/tests/subdir_project.py:92: fine.

sdk/tests/subdir_project.py:91: fine.

sdk/tests/subdir_project.py:90: fine.

sdk/tests/subdir_project.py:89: fine.

sdk/tests/subdir_project.py:88: fine.

sdk/tests/subdir_project.py:87: fine.

sdk/tests/subdir_project.py:86: fine.

sdk/tests/subdir_project.py:85: fine.

sdk/tests/subdir_project.py:84: fine.

sdk/tests/subdir_project.py:83: fine.

sdk/tests/subdir_project.py:82: fine.

sdk/tests/subdir_project.py:81: fine.

sdk/tests/subdir_project.py:80: fine.

sdk/tests/subdir_project.py:79: fine.

sdk/tests/subdir_project.py:78: fine.

sdk/tests/subdir_project.py:77: fine.

sdk/tests/subdir_project.py:76: fine.

sdk/tests/subdir_project.py:75: fine.

sdk/tests/subdir_project.py:74: fine.

sdk/tests/subdir_project.py:73: fine.

sdk/tests/subdir_project.py:72: fine.

sdk/tests/subdir_project.py:71: fine.

sdk/tests/subdir_project.py:70: fine.

sdk/tests/subdir_project.py:69: fine.

sdk/tests/subdir_project.py:68: fine.

sdk/tests/subdir_project.py:67: fine.

sdk/tests/subdir_project.py:66: fine.

sdk/tests/subdir_project.py:65: fine.

sdk/tests/subdir_project.py:64: fine.

sdk/tests/subdir_project.py:63: fine.

sdk/tests/subdir_project.py:62: fine.

sdk/tests/subdir_project.py:61: fine.

sdk/tests/subdir_project.py:60: fine.

sdk/tests/subdir_project.py:59: fine.

sdk/tests/subdir_project.py:58: fine.

sdk/tests/subdir_project.py:57: fine.

sdk/tests/subdir_project.py:56: fine.

sdk/tests/subdir_project.py:55: fine.

sdk/tests/subdir_project.py:54: fine.

sdk/tests/subdir_project.py:53: fine.

sdk/tests/subdir_project.py:52: fine.

sdk/tests/subdir_project.py:51: fine.

sdk/tests/subdir_project.py:50: fine.

sdk/tests/subdir_project.py:49: fine.

sdk/tests/subdir_project.py:48: fine.

sdk/tests/subdir_project.py:47: fine.

sdk/tests/subdir_project.py:46: fine.

sdk/tests/subdir_project.py:45: fine.

sdk/tests/subdir_project.py:44: fine.

sdk/tests/subdir_project.py:43: fine.

sdk/tests/subdir_project.py:42: fine.

sdk/tests/subdir_project.py:41: fine.

sdk/tests/subdir_project.py:40: fine.

sdk/tests/subdir_project.py:39: fine.

sdk/tests/subdir_project.py:38: fine.

sdk/tests/subdir_project.py:37: fine.

sdk/tests/subdir_project.py:36: fine.

sdk/tests/subdir_project.py:35: fine.

sdk/tests/subdir_project.py:34: fine.

sdk/tests/subdir_project.py:33: fine.

sdk/tests/subdir_project.py:32: fine.

sdk/tests/subdir_project.py:31: fine.

sdk/tests/subdir_project.py:30: fine.

sdk/tests/subdir_project.py:29: fine.

sdk/tests/subdir_project.py:28: fine.

sdk/tests/subdir_project.py:27: fine.

sdk/tests/subdir_project.py:26: fine.

sdk/tests/subdir_project.py:25: fine.

sdk/tests/subdir_project.py:24: fine.

sdk/tests/subdir_project.py:23: fine.

sdk/tests/subdir_project.py:22: fine.

sdk/tests/subdir_project.py:21: fine.

sdk/tests/subdir_project.py:20: fine.

sdk/tests/subdir_project.py:19: fine.

sdk/tests/subdir_project.py:18: fine.

sdk/tests/subdir_project.py:17: fine.

sdk/tests/subdir_project.py:16: fine.

sdk/tests/subdir_project.py:15: fine.

sdk/tests/subdir_project.py:14: fine.

sdk/tests/subdir_project.py:13: fine.

sdk/tests/subdir_project.py:12: fine.

sdk/tests/subdir_project.py:11: fine.

sdk/tests/subdir_project.py:10: fine.

sdk/tests/subdir_project.py:9: fine.

sdk/tests/subdir_project.py:8: fine.

sdk/tests/subdir_project.py:7: fine.

sdk/tests/subdir_project.py:6: fine.

sdk/tests/subdir_project.py:5: fine.

sdk/tests/subdir_project.py:4: fine.

sdk/tests/subdir_project.py:3: fine.

sdk/tests/subdir_project.py:2: fine.

sdk/tests/subdir_project.py:1: fine.

sdk/cmake/saturn.cmake:1-304: the only real defects are the missing `SATURN_JANUS_STAGE` global (178, 102-103, 109, 113-114, 216) and the test's wrong reference path (194) and missing `SATURN_JANUS_STAGE` forward (226).