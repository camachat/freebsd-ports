<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	python-gist
DISTVERSIONPREFIX=	v
DISTVERSION=	0.9.2
CATEGORIES=	devel python
=======
PORTNAME=	python-gist
DISTVERSION=	0.10.1
CATEGORIES=	devel python
MASTER_SITES=	CHEESESHOP
>>>>>>> upstream/main
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Command line interface for working with github gists

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE

<<<<<<< HEAD
RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}docopt>0:devel/py-docopt@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}python-gnupg>=0.4.1:security/py-python-gnupg@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}requests>0:www/py-requests@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}simplejson>0:devel/py-simplejson@${PY_FLAVOR}
TEST_DEPENDS=	${PYTHON_PKGNAMEPREFIX}mock>0:devel/py-mock@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}responses>0:devel/py-responses@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}pycodestyle>0:devel/py-pycodestyle@${PY_FLAVOR}

USES=		python:3.6+
USE_GITHUB=	yes
GH_ACCOUNT=	jdowner
GH_PROJECT=	gist
=======
RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}python-gnupg>=0.4.1:security/py-python-gnupg@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}requests>0:www/py-requests@${PY_FLAVOR}
TEST_DEPENDS=	${PYTHON_PKGNAMEPREFIX}mock>0:devel/py-mock@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}responses>0:devel/py-responses@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}pycodestyle>0:devel/py-pycodestyle@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}wheel>0:devel/py-wheel@${PY_FLAVOR}

USES=		python:3.6+
>>>>>>> upstream/main
USE_PYTHON=	autoplist distutils

NO_ARCH=	yes

<<<<<<< HEAD
do-test:
=======
do-test: # tests are currently broken and undergoing redesign, see https://github.com/jdowner/gist/issues/100#issuecomment-817039193
>>>>>>> upstream/main
	@cd ${WRKSRC} && ${PYTHON_CMD} ${PYDISTUTILS_SETUP} test

.include <bsd.port.mk>
