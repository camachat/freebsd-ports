PORTNAME=	selectolax
DISTVERSION=	0.4.7
CATEGORIES=	www python
MASTER_SITES=	PYPI
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	eric@camachat.org
COMMENT=	Fast HTML5 parser with CSS selectors (Lexbor backend only)
WWW=		https://github.com/rushter/selectolax

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	${PYTHON_PKGNAMEPREFIX}setuptools>=0:devel/py-setuptools@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}wheel>=0:devel/py-wheel@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}cython>0:lang/cython@${PY_FLAVOR}

LIB_DEPENDS=	liblexbor.so:www/lexbor

USES=		python
USE_PYTHON=	autoplist distutils

# Remove bundled lexbor submodule that already in ports
post-extract:
	@${RM} -rf ${WRKSRC}/lexbor/lexbor

# Fix license conflict in pyproject.toml and any bundled paths
post-patch:
	@${REINPLACE_CMD} -E 's/^[[:space:]]*license[[:space:]]*=[[:space:]]*.*/license = {file = "LICENSE"}/' ${WRKSRC}/pyproject.toml
	@${REINPLACE_CMD} -E 's|lexbor/lexbor|${LOCALBASE}/include/lexbor|g' ${WRKSRC}/setup.py

.include <bsd.port.mk>
