PORTNAME=	snmpclitools
PORTVERSION=	0.6.1
PORTREVISION=	3
CATEGORIES=	net-mgmt python
MASTER_SITES=	PYPI
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	mhjacks@swbell.net
COMMENT=	Command-line SNMP utilities in Python
WWW=		https://github.com/etingof/snmpclitools

LICENSE=	BSD2CLAUSE

RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}pysnmp>=4.4.4:net-mgmt/py-pysnmp@${PY_FLAVOR}

USES=		python
USE_PYTHON=	concurrent distutils autoplist

NO_ARCH=	yes

.include <bsd.port.mk>
