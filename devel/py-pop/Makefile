# Created by: Kirill Ponomarev <krion@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=		pop
PORTVERSION=	16.1
CATEGORIES=		devel python
=======

PORTNAME=	pop
PORTVERSION=	17.1
CATEGORIES=	devel python
>>>>>>> upstream/main
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	krion@FreeBSD.org
COMMENT=	Plugin Oriented Programming interface for Python

LICENSE=	APACHE20

RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}aiofiles>=0:devel/py-aiofiles@${PY_FLAVOR} \
<<<<<<< HEAD
				${PYTHON_PKGNAMEPREFIX}aiologger>=0:devel/py-aiologger@${PY_FLAVOR} \
				${PYTHON_PKGNAMEPREFIX}msgpack>=0:devel/py-msgpack@${PY_FLAVOR} \
				${PYTHON_PKGNAMEPREFIX}dict-toolbox>=0:sysutils/py-dict-toolbox@${PY_FLAVOR} \
				${PYTHON_PKGNAMEPREFIX}yaml>0:devel/py-yaml@${PY_FLAVOR} \
				${PYTHON_PKGNAMEPREFIX}pop-config>0:devel/py-pop-config@${PY_FLAVOR}

USES=		python:3.6+ shebangfix
USE_PYTHON=	autoplist distutils

USE_GITLAB=	yes
GL_ACCOUNT=	saltstack/pop
GL_COMMIT=	5d00f3b78a0d7caecf7747de7f03985c850d3750
=======
		${PYTHON_PKGNAMEPREFIX}aiologger>=0:devel/py-aiologger@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}dict-toolbox>=0:sysutils/py-dict-toolbox@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}msgpack>=0:devel/py-msgpack@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}pop-config>0:devel/py-pop-config@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}yaml>0:devel/py-yaml@${PY_FLAVOR}

USES=		python:3.6+ shebangfix

USE_GITLAB=	yes
GL_ACCOUNT=	saltstack/pop
GL_COMMIT=	ab9ae13da6087f91edf5465619162141fd1b4e71
USE_PYTHON=	autoplist distutils
>>>>>>> upstream/main

NO_ARCH=	yes

.include <bsd.port.mk>
