# Created by: thierry@pompo.net
<<<<<<< HEAD
# $FreeBSD$
=======
>>>>>>> upstream/main

PORTNAME=	jplephem
PORTVERSION=	2.15
DISTVERSIONPREFIX=	v
<<<<<<< HEAD
=======
PORTREVISION=	1
>>>>>>> upstream/main
CATEGORIES=	astro python
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	thierry@FreeBSD.org
COMMENT=	Python version of NASA DE4xx ephemerides

LICENSE=	MIT

<<<<<<< HEAD
BUILD_DEPENDS=	${NUMPY}
RUN_DEPENDS=	${NUMPY}
=======
BUILD_DEPENDS=	${PYNUMPY}
RUN_DEPENDS=	${PYNUMPY}
>>>>>>> upstream/main
TEST_DEPENDS=	bash:shells/bash	\
		${PYTHON_PKGNAMEPREFIX}unittest2>0:devel/py-unittest2@${PY_FLAVOR}

USE_GITHUB=	yes
GH_ACCOUNT=	brandon-rhodes
GH_PROJECT=	python-jplephem

USES=		python:3.7+ shebangfix
USE_PYTHON=	distutils
SHEBANG_FILES=	test.sh	bin/doctest.sh bin/parse_zzidmap.sh bin/benchmark.py
NO_ARCH=	yes

do-test:	install
	(cd ${WRKSRC} && ./test.sh)

.include <bsd.port.mk>
