# Created by: Kirill Ponomarev <krion@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	tiamat
PORTVERSION=	7.6
=======

PORTNAME=	tiamat
PORTVERSION=	7.7
>>>>>>> upstream/main
CATEGORIES=	devel python
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	krion@FreeBSD.org
COMMENT=	Single binary builder for Python projects

LICENSE=	APACHE20
LICENSE_FILE=	${WRKSRC}/LICENSE

<<<<<<< HEAD
RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}pop>=0:devel/py-pop@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}certifi>=0:security/py-certifi@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}chardet>0:textproc/py-chardet@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}idna>0:dns/py-idna@${PY_FLAVOR} \
=======
RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}certifi>=0:security/py-certifi@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}chardet>0:textproc/py-chardet@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}idna>0:dns/py-idna@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}pop>=0:devel/py-pop@${PY_FLAVOR} \
>>>>>>> upstream/main
		${PYTHON_PKGNAMEPREFIX}requests>=0:www/py-requests@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}urllib3>0:net/py-urllib3@${PY_FLAVOR}

USES=		python:3.6+ shebangfix
<<<<<<< HEAD
USE_PYTHON=	autoplist concurrent distutils

USE_GITLAB=	yes
GL_ACCOUNT=	saltstack/pop
GL_COMMIT=	a6e428a52d2c63590fa49586f9d4cde58f683c44
=======

USE_GITLAB=	yes
GL_ACCOUNT=	saltstack/pop
GL_COMMIT=	5e75fbc37c00cb891287022082c7822a3d300b9c
USE_PYTHON=	autoplist concurrent distutils
>>>>>>> upstream/main

NO_ARCH=	yes

.include <bsd.port.mk>
