"""Настройки gunicorn для recon-service::

    gunicorn -c reconservice/gunicorn_conf.py 'reconservice.app:create_app()'
"""
bind = '0.0.0.0:5560'
workers = 1                # интерактивная сессия и очередь задач — в памяти одного процесса
worker_class = 'gthread'
threads = 8
timeout = 600              # перебор центра на CPU и отдача файлов — секунды и минуты
graceful_timeout = 60      # на остановку: задача получает SIGTERM и удаляет начатый объём
accesslog = None           # без журнала запросов: студия опрашивает сессию и задачу раз в 1–1,5 с (журнал есть у Apache rbtm-web)
errorlog = '-'
loglevel = 'info'


def worker_exit(server, worker):  # noqa: ARG001 — сигнатура хука gunicorn
    """Остановить очередь и сессию: процесс задачи не должен пережить сервис."""
    from reconservice.app import shutdown_all  # noqa: WPS433
    shutdown_all()
