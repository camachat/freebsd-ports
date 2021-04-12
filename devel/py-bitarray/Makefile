# Created by: Sofian Brabez <sbrabez@gmail.com>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	bitarray
PORTVERSION=	1.8.1
=======

PORTNAME=	bitarray
PORTVERSION=	1.9.1
>>>>>>> upstream/main
CATEGORIES=	devel python
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	sbz@FreeBSD.org
COMMENT=	Efficient arrays of booleans

LICENSE=	PSFL
LICENSE_FILE=	${WRKSRC}/LICENSE

USES=		python:3.6+

USE_GITHUB=	yes
USE_PYTHON=	distutils autoplist

GH_ACCOUNT=	ilanschnell
GH_PROJECT=	bitarray

<<<<<<< HEAD
DOCSDIR=	${PREFIX}/share/doc/py-${PORTNAME}
EXAMPLESDIR=	${PREFIX}/share/examples/py-${PORTNAME}

PORTDOCS=	AUTHORS CHANGE_LOG README.md
=======
DOCSDIR=	${PREFIX}/share/doc/${PKGNAMEPREFIX}${PORTNAME}
EXAMPLESDIR=	${PREFIX}/share/examples/${PKGNAMEPREFIX}${PORTNAME}

PORTDOCS=	CHANGE_LOG README.rst TODO changelog.rst
>>>>>>> upstream/main
PORTEXAMPLES=	*

OPTIONS_DEFINE=	DOCS EXAMPLES

post-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${PORTDOCS:S,^,${WRKSRC}/,} ${STAGEDIR}${DOCSDIR}

post-install-EXAMPLES-on:
	@${MKDIR} ${STAGEDIR}${EXAMPLESDIR}
	(cd ${WRKSRC}/examples/ && ${COPYTREE_SHARE} . ${STAGEDIR}${EXAMPLESDIR})

do-test:
	${SETENV} PYTHONPATH=${STAGEDIR}${PYTHONPREFIX_SITELIBDIR} ${PYTHON_CMD} -c "import ${PORTNAME}; ${PORTNAME}.test()"

.include <bsd.port.mk>
