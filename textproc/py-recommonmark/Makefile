PORTNAME=	recommonmark
PORTVERSION=	0.5.0
PORTREVISION=	3
CATEGORIES=	textproc python
MASTER_SITES=	PYPI
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	brooks@FreeBSD.org
COMMENT=	CommonMark bridge for docutils and Sphinx
WWW=		https://github.com/readthedocs/recommonmark

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/license.md

RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}CommonMark>=0.7.3:textproc/py-CommonMark@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}docutils>=0.11:textproc/py-docutils@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}sphinx>=1.3.1,1:textproc/py-sphinx@${PY_FLAVOR}

USES=		python
USE_PYTHON=	autoplist concurrent distutils

NO_ARCH=	yes

.include <bsd.port.mk>
