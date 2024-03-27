import traceback
from logging import Logger
from threading import Lock

from amqp import exceptions as amqp_exceptions
from celery import Celery
from celery import exceptions as celery_exceptions
from celery.result import AsyncResult
from kombu import exceptions as kombu_exceptions

from flower.utils import Singleton
from flower.worker.celery_tasks.celery_config import get_celery_app


class CeleryConnectionError(Exception):
    def __init__(self, connection_address: str, error_details: str):
        message = f"Connection Error: {connection_address=} {error_details=}"
        super().__init__(message)


BAD_APP_OS_ERROR_MESSAGE = "Server unexpectedly closed connection"


class _CeleryWrapperBadAppError(Exception):
    pass


class CeleryTaskTimeoutException(Exception):
    def __init__(
        self, timeout_type: str, connection_address: str, async_result: AsyncResult
    ):
        message = (
            f"Timeout Exception: {timeout_type=} {connection_address=} while waiting "
            f"for {async_result.task_id=}"
        )
        super().__init__(message)


class CeleryWrapper:
    def __init__(self, socket_addr: str):
        self._socket_addr = socket_addr
        self._change_app_lock = Lock()
        self._celery_app = self._instantiate_celery_object()

    def _reset_celery_app(self):
        with self._change_app_lock:
            old_app = self._celery_app
            self._celery_app = self._instantiate_celery_object()
            old_app.close()

    # queue_task() is non-blocking, because apply_async() is non-blocking
    def get(
        self,
        task_signature: str,
        logger: Logger,
        priority=None,
        *args,
        **kwargs,
    ):
        async_result = self._queue_task(task_signature, logger, priority, *args, **kwargs)
        return self._get_result(async_result,60, logger)

    def _queue_task(
        self,
        task_signature: str,
        logger: Logger,
        priority=None,
        *args,
        **kwargs,
    ) -> AsyncResult:
        try:
            task_signature = self._celery_app.signature(task_signature)
            return task_signature.apply_async(args, kwargs, priority=priority)
        except (
            kombu_exceptions.OperationalError,
            amqp_exceptions.AccessRefused,
            amqp_exceptions.NotAllowed,
        ) as exc:
            logger.error(
                f"{self._socket_addr=} Exception: {exc=} was caught. Most likely this "
                f"means the broker is not accessible. Queuing of {task_signature=} with "
                f"{args=} and {kwargs=} failed."
            )

            # The celery app needs to be recreated due to a bug:
            # https://github.com/celery/celery/issues/6912
            # If we don't reset the celery app queuing a task will work, but then we won't be able to fetch its result.
            self._reset_celery_app()

            raise CeleryConnectionError(
                connection_address=self._socket_addr,
                error_details=f"While queuing {task_signature=} with {args=} and {kwargs=} on {self._socket_addr=}.",
            )
        except Exception as exc:
            logger.error(traceback.format_exc())
            raise exc

    @staticmethod
    def _get_result_with_os_error_handler(async_result, timeout):
        """
        When the consumer of a celery app gets corrupted due to a rabbitmq restart,
        the error thrown is an OSError. The problem with that is that this error is
        a generic one and many other (e.g. PermissionError) inherit from it.
        We need to make sure that it's the OSError we expect, it includes the
        BAD_APP_OS_ERROR_MESSAGE, in order to reset the celery_app.
        """
        try:
            return async_result.get(timeout)
        except OSError as exc:
            if BAD_APP_OS_ERROR_MESSAGE == str(exc):
                raise _CeleryWrapperBadAppError()
            else:
                raise exc

    # get_result() is blocking, because celery.result.AsyncResult.get() is blocking
    def _get_result(self, async_result: AsyncResult, timeout: int, logger: Logger):
        try:
            return self._get_result_with_os_error_handler(async_result, timeout)
        except (
            _CeleryWrapperBadAppError,
            amqp_exceptions.UnexpectedFrame,
            ConnectionResetError,
            ConnectionRefusedError,
            kombu_exceptions.OperationalError,
        ) as exc:
            logger.error(
                f"{self._socket_addr=} Exception: {exc=} was raised. Most likely this "
                f"means that the rabbitmq is down."
            )

            # The celery app needs to be recreated due to a bug:
            # https://github.com/celery/celery/issues/6912
            # If we don't reset the celery app queuing a task will work, but then we won't be able to fetch its result.
            self._reset_celery_app()

            raise CeleryConnectionError(
                connection_address=self._socket_addr,
                error_details=f"While getting {async_result.id=}.",
            )
        except celery_exceptions.TimeoutError as timeout_error:
            raise CeleryTaskTimeoutException(
                timeout_type=str(type(timeout_error)),
                connection_address=self._socket_addr,
                async_result=async_result,
            )
        except Exception as exc:
            logger.error(traceback.format_exc())
            raise exc

    # It seems that Celery objects are somewhat expensive, do not have more than one
    # instance per node at any time
    def _instantiate_celery_object(self) -> Celery:
        user = "user"
        password = "password"
        vhost = "user_vhost"
        return get_celery_app(user, password, self._socket_addr, vhost)


class CeleryAppFactory(metaclass=Singleton):
    def __init__(self):
        self._celery_apps = {}

    def get_celery_app(self, socket_addr: str) -> CeleryWrapper:
        if socket_addr in self._celery_apps:
            return self._celery_apps[socket_addr]
        else:
            self._celery_apps[socket_addr] = CeleryWrapper(socket_addr)
            return self._celery_apps[socket_addr]
