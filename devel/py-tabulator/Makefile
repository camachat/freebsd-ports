PORTNAME=	tabulator
PORTVERSION=	1.53.5
PORTREVISION=	3
CATEGORIES=	devel python
MASTER_SITES=	PYPI
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	sunpoet@FreeBSD.org
COMMENT=	Consistent interface for stream reading and writing tabular data (csv/xls/json/etc)
WWW=		https://github.com/frictionlessdata/tabulator-py

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE.md

BUILD_DEPENDS=	${PYTHON_PKGNAMEPREFIX}setuptools>=0:devel/py-setuptools@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}wheel>=0:devel/py-wheel@${PY_FLAVOR}
RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}boto3>=1.9:www/py-boto3@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}chardet>=3.0,1:textproc/py-chardet@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}click>=6.0:devel/py-click@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}ijson>=3.0.3:devel/py-ijson@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}jsonlines>=1.1:devel/py-jsonlines@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}linear-tsv>=1.0:devel/py-linear-tsv@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}openpyxl>=2.6:textproc/py-openpyxl@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}requests>=2.8:www/py-requests@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}six>=1.9:devel/py-six@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}sqlalchemy20>=0.9.6:databases/py-sqlalchemy20@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}unicodecsv>=0.14:devel/py-unicodecsv@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}xlrd>=1.0:textproc/py-xlrd@${PY_FLAVOR}

USES=		python
USE_PYTHON=	autoplist concurrent pep517

NO_ARCH=	yes

OPTIONS_DEFINE=	CCHARDET DATAPACKAGE
CCHARDET_DESC=		Character encoding detection via cchardet
DATAPACKAGE_DESC=	Data Package support

CCHARDET_RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}cchardet>=2.0:textproc/py-cchardet@${PY_FLAVOR}
DATAPACKAGE_RUN_DEPENDS=${PYTHON_PKGNAMEPREFIX}datapackage>=1.12:devel/py-datapackage@${PY_FLAVOR}

.include <bsd.port.mk>
