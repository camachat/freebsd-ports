PORTNAME=	nose
PORTVERSION=	1.3.7
PORTREVISION=	2
CATEGORIES=	devel python
MASTER_SITES=	PYPI
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	bofh@FreeBSD.org
COMMENT=	Discovery-based unittest extension
WWW=		https://nose.readthedocs.io/

LICENSE=	LGPL21
LICENSE_FILE=	${WRKSRC}/lgpl.txt

USES=		python
USE_PYTHON=	autoplist concurrent distutils

NO_ARCH=	yes

do-test:
# Extra tests for python 3, added in by setup3lib.py
	@cd ${WRKSRC} && ${PYTHON_CMD} ${PYDISTUTILS_SETUP} build_tests
	@cd ${WRKSRC} && ${PYTHON_CMD} selftest.py

.include <bsd.port.mk>
