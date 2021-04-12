<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	asgiref
PORTVERSION=	3.2.10
=======
PORTNAME=	asgiref
PORTVERSION=	3.3.2
>>>>>>> upstream/main
CATEGORIES=	www python
MASTER_SITES=	CHEESESHOP
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	wen@FreeBSD.org
COMMENT=	ASGI specs, helper code, and adapters

LICENSE=	BSD3CLAUSE
LICENSE_FILE=	${WRKSRC}/LICENSE

TEST_DEPENDS=	${PYTHON_PKGNAMEPREFIX}pytest>=4.3.0:devel/py-pytest@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}pytest-asyncio>=0.10.0:devel/py-pytest-asyncio@${PY_FLAVOR}

USES=		python:3.5+
USE_PYTHON=	distutils autoplist

NO_ARCH=	yes

do-test:
	(cd ${WRKSRC} && pytest)

.include <bsd.port.mk>
