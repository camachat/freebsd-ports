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
		${PYTHON_PKGNAMEPREFIX}cython>0:lang/cython@${PY_FLAVOR} \
		${LOCALBASE}/bin/cmake:devel/cmake-core

USES=		gmake python
USE_PYTHON=	autoplist distutils
CFLAGS+=	-Wno-void-pointer-to-enum-cast

PYDISTUTILS_CONFIGUREARGS+=	--static --lexbor --cython
PYDISTUTILS_BUILDARGS+=	--static --lexbor --cython
PYDISTUTILS_INSTALLARGS+=	--static --lexbor --cython

# Fix license conflict in pyproject.toml and any bundled paths
post-patch:
	@${REINPLACE_CMD} -E 's/^[[:space:]]*license[[:space:]]*=[[:space:]]*.*/license = {file = "LICENSE"}/' ${WRKSRC}/pyproject.toml
	@${REINPLACE_CMD} -E 's|libmodest_static.a|libmodest.a|g' ${WRKSRC}/setup.py
pre-configure:
	cd ${WRKSRC}/modest && cmake . -DCMAKE_POSITION_INDEPENDENT_CODE=ON && cmake --build . -j ${MAKE_JOBS_NUMBER}
	cd ${WRKSRC}/lexbor && cmake . -DCMAKE_POSITION_INDEPENDENT_CODE=ON && cmake --build . -j ${MAKE_JOBS_NUMBER}
	cd ${WRKSRC}/modest && cmake --install . --prefix .
	cd ${WRKSRC}/lexbor &&cmake --install . --prefix .

.include <bsd.port.mk>
