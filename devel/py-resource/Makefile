PORTNAME=	resource
PORTVERSION=	0.2.1
PORTREVISION=	2
CATEGORIES=	devel python
MASTER_SITES=	PYPI
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}
DISTNAME=	Resource-${PORTVERSION}

MAINTAINER=	bofh@FreeBSD.org
COMMENT=	Library concentrated on the Resource layer of RESTful APIs
WWW=		https://github.com/RussellLuo/resource

LICENSE=	MIT

RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}jsonform>=0.0.2:devel/py-jsonform@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}jsonsir>=0.0.2:devel/py-jsonsir@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}python-easyconfig>=0.1.0:devel/py-python-easyconfig@${PY_FLAVOR}

USES=		python
USE_PYTHON=	autoplist distutils

.include <bsd.port.mk>
