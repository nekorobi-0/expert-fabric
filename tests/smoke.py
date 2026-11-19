import math
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


def run(binary, commands, *options):
    return subprocess.run(
        [binary, *options], input=commands, text=True, capture_output=True, timeout=10
    )


def exercise_engine(binary, directory):
    weights1 = directory / 'a weights.txt'
    weights1.write_text('EFMLP1 1 1\n1 1 1\n')
    weights2 = directory / 'second.txt'
    weights2.write_text('EFMLP1 1 1\n1 1 2\n')
    broken = directory / 'broken.txt'
    broken.write_text('EFMLP1 1 1\n1 1\n')
    config = directory / 'capacity.yml'
    config.write_text('expert_engines: [{transport: local, target_kind: cpu, capacity_bytes: 24}]\n')
    result = run(binary, f'''HELLO
LOAD ffn 1 "{broken}"
HELLO
LOAD ffn 1 "{weights1}"
INVOKE ffn 1 2
PUBLISH ffn 1 -
INVOKE ffn 1 2 -1
LOAD ffn 2 "{weights2}"
LOAD ffn 3 "{weights2}"
PUBLISH ffn 2 999
PUBLISH ffn 2 1
INVOKE ffn 1 2
INVOKE ffn 2 2
RETIRE ffn 1
HELLO
LOAD ffn 1 "{weights1}"
STATUS ffn 1
STATUS ffn 2
STATUS ffn 999
QUIT
''', '--config', str(config))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 19, lines
    assert lines[0] == lines[2] == 'OK CPU max_memory_bytes=24 available_memory_bytes=24'
    assert lines[1].startswith('ERR LOAD_FAILED:'), lines
    assert lines[3] == 'OK READY'
    assert lines[4].startswith('ERR NOT_READY:'), lines
    assert lines[5] == 'OK ACTIVE'
    first = list(map(float, lines[6].split()[1:]))
    expected = [x * x / (1 + math.exp(-x)) for x in (2, -1)]
    assert len(first) == 2 and all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(first, expected))
    assert lines[7] == 'OK READY'
    assert lines[8].startswith('ERR NO_CAPACITY:'), lines
    assert lines[9].startswith('ERR STALE_GENERATION:'), lines
    assert lines[10] == 'OK ACTIVE'
    assert lines[11].startswith('ERR STALE_GENERATION:'), lines
    assert math.isclose(float(lines[12].split()[1]), expected[0] * 2, rel_tol=1e-6)
    assert lines[13] == 'OK'
    assert lines[14] == 'OK CPU max_memory_bytes=24 available_memory_bytes=12'
    assert lines[15].startswith('ERR STALE_GENERATION:'), lines
    assert lines[16:] == ['OK RETIRED', 'OK ACTIVE', 'OK ABSENT'], lines


def exercise_exporter(binary, directory):
    config = directory / 'metrics.yml'
    config.write_text('''profiling:
  prometheus: {enabled: true, listen: '127.0.0.1:0'}
  per_expert_counters: true
expert_engines: [{transport: local, target_kind: cpu}]
''')
    with subprocess.Popen([binary, '--config', str(config)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            from select import select
            assert select([process.stderr], [], [], 5)[0], 'metrics endpoint never became ready'
            endpoint = re.fullmatch(r'Metrics: (http://127\.0\.0\.1:\d+)/metrics\n', process.stderr.readline())
            assert endpoint, 'metrics endpoint not announced'
            process.stdin.write(f'LOAD ffn 1 "{directory / "a weights.txt"}"\n'
                                'PUBLISH ffn 1 -\nINVOKE ffn 1 1 2\n')
            process.stdin.flush()
            assert select([process.stdout], [], [], 5)[0], 'expert commands did not complete'
            answers = [process.stdout.readline().strip() for _ in range(3)]
            assert answers[:2] == ['OK READY', 'OK ACTIVE'] and answers[2].startswith('OK '), answers
            with urllib.request.urlopen(endpoint[1] + '/metrics', timeout=3) as connection:
                text = connection.read().decode()
            assert 'expert_fabric_stage_duration_seconds_count{stage="expert_compute"} 1' in text, text
            assert 'expert_fabric_expert_invocations_total{expert_id="ffn"} 1' in text, text
            assert 'expert_fabric_expert_tokens_total{expert_id="ffn"} 2' in text, text
            try:
                urllib.request.urlopen(endpoint[1] + '/missing', timeout=3)
            except urllib.error.HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError('unexpected HTTP success for unknown path')
            process.stdin.write('QUIT\n')
            process.stdin.flush()
            assert process.wait(timeout=5) == 0
        finally:
            if process.poll() is None:
                process.kill()


def exercise_config(binary, directory):
    config = directory / 'invalid.yml'
    config.write_text('expert_engines: [{transport: local, transport: upi}]\n')
    duplicate = run(binary, '', '--config', str(config))
    assert duplicate.returncode != 0 and 'duplicate key' in duplicate.stderr, duplicate.stderr
    config.write_text('expert_engines: [{transport: roce_v2, target_kind: cpu}]\n')
    rdma = run(binary, '', '--config', str(config))
    assert rdma.returncode != 0 and '.endpoint: required for roce_v2' in rdma.stderr, rdma.stderr
    config.write_text('expert_engines: [{transport: local, source_address: "192.0.2.1"}]\n')
    local = run(binary, '', '--config', str(config))
    assert local.returncode != 0 and '.source_address: requires a RoCE v2 engine' in local.stderr, local.stderr


def exercise_placement(binary, directory, second_kind):
    config = directory / 'placement.yml'
    config.write_text(f'''expert_engines:
  - {{id: first, target_kind: cpu, capacity_bytes: 12, bandwidth_bytes_per_second: 20}}
  - {{id: second, target_kind: {second_kind}, capacity_bytes: 12, bandwidth_bytes_per_second: 10}}
''')
    weights = directory / 'a weights.txt'
    result = run(binary, f'''ENGINES
AUTOLOAD first-expert 1 "{weights}"
AUTOLOAD next-expert 1 "{weights}"
AUTOLOAD third-expert 1 "{weights}"
@second PUBLISH next-expert 1 -
@second INVOKE next-expert 1 2
@first STATUS next-expert 1
''', '--config', str(config))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert 'first local OK CPU max_memory_bytes=12' in lines[0], lines
    assert f'second local OK {second_kind.upper()} max_memory_bytes=12' in lines[0], lines
    assert lines[1:3] == ['OK READY @first', 'OK READY @second'], lines
    assert lines[3].startswith('ERR NO_CAPACITY:'), lines
    assert lines[4] == 'OK ACTIVE', lines
    assert math.isclose(float(lines[5].split()[1]), 4 / (1 + math.exp(-2)), rel_tol=1e-6), lines
    assert lines[6] == 'OK ABSENT', lines


def exercise_ngram(binary, directory):
    config = directory / 'ngram.yml'
    config.write_text('''speculation:
  ngram: {enabled: true, min_match_tokens: 2, max_match_tokens: 4, cache: {max_ram_mib: 1}}
expert_engines: [{}]
''')
    result = run(binary, '''NGRAM_PROPOSE model tokenizer session 1 2
NGRAM_OBSERVE model tokenizer session 1 2 3 4 5
NGRAM_PROPOSE model tokenizer session 9 1 2
NGRAM_PROPOSE model tokenizer other 1 2
NGRAM_PROPOSE model tokenizer session -1
''', '--config', str(config))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:4] == ['OK', 'OK', 'OK 3 4 5', 'OK'], lines
    assert lines[4].startswith('ERR INVALID_ARGUMENT:'), lines


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as scratch:
        workdir = Path(scratch)
        exercise_engine(sys.argv[1], workdir)
        exercise_placement(sys.argv[1], workdir, 'gpu' if '--hip' in sys.argv[2:] else 'cpu')
        exercise_ngram(sys.argv[1], workdir)
        exercise_config(sys.argv[1], workdir)
        exercise_exporter(sys.argv[1], workdir)
    print('Expert engine, placement, n-gram, config, and exporter smoke tests passed')
