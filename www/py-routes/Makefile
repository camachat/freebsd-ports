PORTNAME=	routes
PORTVERSION=	2.5.1
PORTREVISION=	1
CATEGORIES=	www python
MASTER_SITES=	PYPI/source/R/Routes
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}
DISTNAME=	Routes-${PORTVERSION}

MAINTAINER=	wen@FreeBSD.org
COMMENT=	Routing package for Python that matches URLs to dicts and vice versa
WWW=		https://routes.readthedocs.io/

LICENSE=	MIT

RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}six>0:devel/py-six@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}repoze.lru>=0.3:devel/py-repoze.lru@${PY_FLAVOR}

USES=		python
USE_PYTHON=	autoplist distutils

NO_ARCH=	yes

.include <bsd.port.mk>
