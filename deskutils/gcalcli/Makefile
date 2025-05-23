PORTNAME=	gcalcli
PORTVERSION=	4.0.4
DISTVERSIONPREFIX=	v
PORTREVISION=	4
CATEGORIES=	deskutils python
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	ports@thelanman.net
COMMENT=	Google Calendar Command Line Interface
WWW=		https://github.com/insanum/gcalcli

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE

DEPRECATED=	Depends on expired security/py-oauth2client
EXPIRATION_DATE=2025-06-01

RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}python-dateutil>=0:devel/py-python-dateutil@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}google-api-python-client>=1.4:www/py-google-api-python-client@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}httplib2>=0:www/py-httplib2@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}parsedatetime>=0:devel/py-parsedatetime@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}oauth2client>0:security/py-oauth2client@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}six>0:devel/py-six@${PY_FLAVOR}

USES=		python
USE_PYTHON=	autoplist concurrent distutils

USE_GITHUB=	yes
GH_ACCOUNT=	insanum

NO_ARCH=	yes

OPTIONS_DEFINE=		ICS_VCAL
OPTIONS_DEFAULT=	ICS_VCAL

ICS_VCAL_DESC=		Enable importing of ics/vcal files
ICS_VCAL_RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}vobject>=0.4.8:deskutils/py-vobject@${PY_FLAVOR}

.include <bsd.port.mk>
