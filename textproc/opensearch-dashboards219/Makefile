PORTNAME=	opensearch-dashboards
DISTVERSION=	2.19.1
DISTVERSIONSUFFIX=	-linux-${OPENSEARCH_ARCH}
PORTREVISION=	1
CATEGORIES=	textproc www
MASTER_SITES=	https://artifacts.opensearch.org/releases/bundle/${PORTNAME}/${DISTVERSION}/
PKGNAMESUFFIX=	219

MAINTAINER=	opensearch@FreeBSD.org
COMMENT=	Visualization dashboards for OpenSearch
WWW=		https://opensearch.org

LICENSE=	APACHE20
LICENSE_FILE=	${WRKSRC}/LICENSE.txt

ONLY_FOR_ARCHS=	aarch64 amd64

CONFLICTS=	opensearch-dashboards opensearch-dashboards210 opensearch-dashboards213 \
		opensearch-dashboards216

BUILD_DEPENDS=	npm-node20>0:www/npm-node20

USES=		nodejs:20,build,run python:build
USE_RC_SUBR=	${PORTNAME}

SUB_FILES=	pkg-deinstall
SUB_LIST=	PORTNAME="${PORTNAME}" \
		PKGNAMESUFFIX="${PKGNAMESUFFIX}" \
		WWWDIR=${WWWDIR}

WRKSRC=		${WRKDIR}/${PORTNAME}-${PORTVERSION}

NODE_VER=	20.19.2
NO_BUILD=	yes

_DEVDIR=	${WRKDIR}/.devdir

.include <bsd.port.pre.mk>

.if ${ARCH} == "aarch64"
OPENSEARCH_ARCH=	arm64
.else
OPENSEARCH_ARCH=	x64
.endif

post-patch:
	@${REINPLACE_CMD} -e 's|<19|<=${NODE_VER}|g' ${WRKSRC}/package.json
	${RM} \
		${WRKSRC}/bin/*.orig \
		${WRKSRC}/bin/*.bak \
		${WRKSRC}/package.json.bak
	${RM} -r ${WRKSRC}/plugins/reportsDashboards/.chromium

do-install:
	${MKDIR} ${STAGEDIR}${WWWDIR} ${STAGEDIR}${ETCDIR}
	${INSTALL_DATA} ${WRKSRC}/config/opensearch_dashboards.yml ${STAGEDIR}${ETCDIR}/opensearch_dashboards.yml.sample
	(cd ${WRKSRC} && \
		${COPYTREE_SHARE} "data node_modules package.json plugins src" ${STAGEDIR}${WWWDIR} && \
		${COPYTREE_BIN} bin ${STAGEDIR}${WWWDIR})

post-install:
	${ECHO} "@sample ${ETCDIR}/opensearch_dashboards.yml.sample" >> ${TMPPLIST}
	${FIND} -s ${STAGEDIR}${WWWDIR} -not -type d | ${SORT} | \
		${SED} -e 's#^${STAGEDIR}${PREFIX}/##' >> ${TMPPLIST}
	${ECHO} "@dir(www,www) ${WWWDIR}/data" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/reportsDashboards/node_modules/set-interval-async/test/resources/legacy" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/reportsDashboards/node_modules/set-interval-async/test/resources/fixed" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/reportsDashboards/node_modules/set-interval-async/test/resources/dynamic" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/reportsDashboards/node_modules/set-interval-async/examples/fixed" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/reportsDashboards/node_modules/set-interval-async/examples/dynamic" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/observabilityDashboards/server/adaptors/integrations/__test__" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/observabilityDashboards/server/adaptors/integrations/repository/__test__" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Alerts/components/AlertFlyout" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/CreateDetector/components/ConfigureAlerts/components/AlertCondition" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/components/AlertTriggerView" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/components/DetectorBasicDetailsView" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/components/DetectorRulesView" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/components/FieldMappingsView" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/components/UpdateAlertConditions" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/components/UpdateDetectorBasicDetails" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/components/UpdateDetectorRules" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/components/UpdateFieldMappings" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/containers/AlertTriggersView" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/containers/DetectorDetails" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/containers/DetectorDetailsView" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/containers/Detectors" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Detectors/containers/EditFieldMappings" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Rules/components/DeleteModal" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Rules/components/RuleEditor/components" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityAnalyticsDashboards/test/mocks/Rules/components/RulesTable" >> ${TMPPLIST}
	${ECHO} "@dir ${WWWDIR}/plugins/securityDashboards/server/multitenancy/test" >> ${TMPPLIST}

.include <bsd.port.post.mk>
