PORTNAME=	openssl
PORTVERSION=	25.0.0
PORTREVISION=	1
PORTEPOCH=	1
CATEGORIES=	security python
MASTER_SITES=	PYPI
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}
DISTNAME=	pyopenssl-${PORTVERSION}

MAINTAINER=	sbz@FreeBSD.org
COMMENT=	Python interface to the OpenSSL library
WWW=		https://github.com/pyca/pyopenssl

LICENSE=	APACHE20
LICENSE_FILE=	${WRKSRC}/LICENSE

RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}typing-extensions>=4.9:devel/py-typing-extensions@${PY_FLAVOR}
TEST_DEPENDS=	${PYTHON_PKGNAMEPREFIX}pretend>0:devel/py-pretend@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}pytest-rerunfailures>0:devel/py-pytest-rerunfailures@${PY_FLAVOR}

USES=		cpe python
USE_PYTHON=	autoplist concurrent cryptography distutils pytest pythonprefix

NO_ARCH=	yes

CPE_VENDOR=	jean-paul_calderone
CPE_PRODUCT=	py${PORTNAME}

# These tests will be skipped due the following reasons:
#
# * test_connect_refused: Avoid stalling in build environments that don't have
#   a internet connection
# * test_set_verify_callback_exception: Fails on *BSD systems, see also
#   https://github.com/pyca/pyopenssl/issues/656 for further details
# * test_connect_ex: succeeds unexpectedly:
#   https://github.com/pyca/pyopenssl/issues/1192
#
# The system clock must be in sync otherwise the test "test_verify_with_time"
# will fail.
PYTEST_IGNORED_TESTS=	test_connect_ex \
			test_connect_refused \
			test_set_verify_callback_exception
TEST_ENV=	LC_ALL=en_US.UTF-8 \
		PYTHONPATH=${STAGEDIR}${PYTHONPREFIX_SITELIBDIR}

.include <bsd.port.mk>
