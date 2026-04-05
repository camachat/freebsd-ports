# $FreeBSD$

PORTNAME=	diamcp
DISTVERSION=	20260403
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}
CATEGORIES=	misc python

MAINTAINER=	eric@camachat.org
COMMENT=	All-in-one MCP server for llama.cpp webui with many built-in tools
WWW=		https://github.com/chartrambiz/diaMCP

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE

RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}mcp>=0:misc/py-mcp@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}ddgs>=0:www/py-ddgs@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}httpx>=0:www/py-httpx@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}selectolax>=0:www/py-selectolax@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}pydantic2>=0:devel/py-pydantic2@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}yfinance>=0:finance/py-yfinance@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}requests>=0:www/py-requests@${PY_FLAVOR}

USES=		python
USE_GITHUB=	yes
GH_ACCOUNT=	chartrambiz
GH_PROJECT=	diaMCP
GH_TAGNAME=	e736513

NO_BUILD=	yes

SUB_FILES=	diamcp
SUB_LIST=	PYTHON_CMD=${PYTHON_CMD} \
		PYTHON_SITELIBDIR=${PYTHON_SITELIBDIR} \
		PREFIX=${PREFIX}

do-install:
	@${MKDIR} ${STAGEDIR}${PYTHON_SITELIBDIR}/diamcp
	@${MKDIR} ${STAGEDIR}${PYTHON_SITELIBDIR}/diamcp/tools

	# Install Python files
	${INSTALL_DATA} ${WRKSRC}/*.py ${STAGEDIR}${PYTHON_SITELIBDIR}/diamcp/
	${INSTALL_DATA} ${WRKSRC}/requirements.txt ${STAGEDIR}${PYTHON_SITELIBDIR}/diamcp/

	# Install tools directory
	${CP} -R ${WRKSRC}/tools ${STAGEDIR}${PYTHON_SITELIBDIR}/diamcp/

	# Install rc script
	${INSTALL_SCRIPT} ${WRKDIR}/diamcp ${STAGEDIR}${PREFIX}/etc/rc.d/diamcp

.include <bsd.port.mk>
