"""Тесты общей части recon-service: токен, exp_id, бинарный формат, /health."""
import os

import numpy as np
import pytest

from reconservice import binary
from reconservice.config import Config
from service_helpers import HEADERS, TOKEN, make_service


def test_config_from_env():
    cfg = Config.from_env({'RECON_TOKEN': 't', 'RECON_JOB_GPU': '0', 'RECON_LEGACY_QUEUE': 'false',
                           'RECON_SESSION_TTL_S': '60', 'RECON_FAST': '/ssd'})
    assert cfg.token == 't' and cfg.job_gpu == '0' and cfg.legacy_queue is False
    assert cfg.session_ttl_s == 60.0
    # кэш студии — не внутри /fast/<id>/ (его ноутбук копирует в хранилище и удаляет)
    assert os.path.normpath(cfg.fast_exp_dir('e1')) == os.path.normpath('/ssd/studio/e1')


def test_health_is_public_and_reports_state(tmp_path):
    _, client, _ = make_service(tmp_path)
    r = client.get('/health')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['token_configured'] is True
    assert 'session' in body and 'jobs' in body


@pytest.mark.parametrize('hdrs, code', [({}, 403), ({'X-Recon-Token': 'wrong'}, 403), (HEADERS, 404)])
def test_token_required(tmp_path, hdrs, code):
    _, client, _ = make_service(tmp_path)
    r = client.get('/scans/no-such-exp/info', headers=hdrs)
    assert r.status_code == code


def test_service_without_token_refuses(tmp_path):
    _, client, _ = make_service(tmp_path, token='')
    assert client.get('/scans/x/info', headers={'X-Recon-Token': ''}).status_code == 503
    assert client.get('/health').status_code == 200


@pytest.mark.parametrize('exp_id', ['..', '.hidden', 'a/b', 'a..b', 'x' * 200])
def test_bad_exp_id_rejected(tmp_path, exp_id):
    _, client, _ = make_service(tmp_path)
    r = client.get('/scans/{}/info'.format(exp_id), headers=HEADERS)
    assert r.status_code in (400, 404)       # «a/b» не совпадает с маршрутом — 404, остальные — 400
    if '/' not in exp_id:
        assert r.status_code == 400


def test_binary_roundtrip_quantized_and_raw():
    a = np.linspace(-1, 3, 12 * 20, dtype='float32').reshape(12, 20)
    from flask import Flask
    with Flask(__name__).app_context():
        r = binary.array_response(a, lo=-1.0, hi=3.0, meta={'угол': 1.5})
        back, meta = binary.decode(r.get_data(), dict(r.headers))
        assert back.shape == a.shape and np.abs(back - a).max() <= (4.0 / 65535) * 0.51
        assert meta == {'угол': 1.5}
        assert r.headers['X-Meta'].isascii()
        r = binary.array_response(a, quantized=False)
        back, _ = binary.decode(r.get_data(), dict(r.headers))
        assert back.dtype == np.float32 and np.array_equal(back, a)
        r = binary.array_response(np.ones((3000, 1000), 'float32'), max_px=1400)
        back, meta = binary.decode(r.get_data(), dict(r.headers))
        assert back.shape == (1000, 333) and meta['downsample'] == 3      # ceil(3000 / 1400) = 3
