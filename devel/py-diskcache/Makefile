PORTNAME=	diskcache
DISTVERSIONPREFIX=	v
DISTVERSION=	5.6.3
PORTREVISION=	2
CATEGORIES=	devel python
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

PATCH_SITES=	https://github.com/${GH_ACCOUNT}/${GH_PROJECT}/commit/
PATCHFILES+=	b89111d.patch:-p1 # https://github.com/grantjenks/python-diskcache/pull/311

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Disk and file backed persistent cache
WWW=		https://grantjenks.com/docs/diskcache/

LICENSE=	APACHE20

PY_DEPENDS=	${PYTHON_PKGNAMEPREFIX}django42>0:www/py-django42@${PY_FLAVOR}
BUILD_DEPENDS=	${PY_DEPENDS}
RUN_DEPENDS=	${PY_DEPENDS}

USES=		python
USE_PYTHON=	distutils autoplist pytest

USE_GITHUB=	yes
GH_ACCOUNT=	grantjenks
GH_PROJECT=	python-diskcache

TEST_ENV=	PYTHONPATH=${STAGEDIR}${PYTHONPREFIX_SITELIBDIR}

NO_ARCH=	yes

.include <bsd.port.mk>
