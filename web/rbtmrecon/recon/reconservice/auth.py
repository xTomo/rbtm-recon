"""Проверка запросов: общий токен, пользователь, идентификатор эксперимента."""
from __future__ import annotations

import hmac
import re

from flask import abort, current_app, jsonify, make_response, request

TOKEN_HEADER = 'X-Recon-Token'
USER_HEADER = 'X-Recon-User'
#: exp_id попадает в пути файлов: только буквы, цифры, «.», «_», «-», без «..» и ведущей точки
EXP_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
#: пути без токена (проверка живости контейнера)
PUBLIC_PATHS = frozenset({'/health'})


def _fail(status: int, message: str):
    abort(make_response(jsonify({'error': message}), status))


def require_token() -> None:
    """before_request: токен из ``X-Recon-Token`` должен совпасть с ``RECON_TOKEN``.

    Если токен в сервисе не задан — всё, кроме ``/health``, отвечает 503 (сервис не должен работать открытым)."""
    if request.path in PUBLIC_PATHS:
        return
    expected = current_app.config['RECON'].token
    if not expected:
        _fail(503, 'RECON_TOKEN не задан — сервис не принимает запросы')
    got = request.headers.get(TOKEN_HEADER, '')
    if not hmac.compare_digest(got.encode('utf8'), expected.encode('utf8')):
        _fail(403, 'неверный токен')


def current_user() -> str:
    """Пользователь rbtm-web, от имени которого пришёл запрос (пустая строка — не указан)."""
    return request.headers.get(USER_HEADER, '').strip()[:150]


def valid_exp_id(exp_id: str) -> str:
    if not isinstance(exp_id, str) or not EXP_ID_RE.match(exp_id) or '..' in exp_id:
        _fail(400, 'некорректный exp_id')
    return exp_id
