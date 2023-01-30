/*
 * DBeaver - Universal Database Manager
 * Copyright (C) 2010-2022 DBeaver Corp and others
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.jkiss.dbeaver.ext.hana.model;

import org.jkiss.code.NotNull;
import org.jkiss.code.Nullable;
import org.jkiss.dbeaver.DBException;
import org.jkiss.dbeaver.Log;
import org.jkiss.dbeaver.ext.generic.model.*;
import org.jkiss.dbeaver.ext.generic.model.meta.GenericMetaModel;
import org.jkiss.dbeaver.ext.generic.model.meta.GenericMetaObject;
import org.jkiss.dbeaver.model.DBPDataSourceContainer;
import org.jkiss.dbeaver.model.DBPErrorAssistant;
import org.jkiss.dbeaver.model.DBPEvaluationContext;
import org.jkiss.dbeaver.model.DBUtils;
import org.jkiss.dbeaver.model.exec.jdbc.JDBCPreparedStatement;
import org.jkiss.dbeaver.model.exec.jdbc.JDBCResultSet;
import org.jkiss.dbeaver.model.exec.jdbc.JDBCSession;
import org.jkiss.dbeaver.model.exec.jdbc.JDBCStatement;
import org.jkiss.dbeaver.model.impl.jdbc.JDBCConstants;
import org.jkiss.dbeaver.model.impl.jdbc.JDBCUtils;
import org.jkiss.dbeaver.model.runtime.DBRProgressMonitor;
import org.jkiss.dbeaver.model.sql.format.SQLFormatUtils;
import org.jkiss.dbeaver.model.struct.DBSEntityConstraintType;
import org.jkiss.dbeaver.model.struct.rdb.DBSProcedureType;
import org.jkiss.utils.CommonUtils;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * HANAMetaModel
 */
public class HANAMetaModel extends GenericMetaModel
{
    private static final Log log = Log.getLog(HANAMetaModel.class);
    private static Pattern ERROR_POSITION_PATTERN = Pattern.compile(" \\(at pos ([0-9]+)\\)");
    
    public HANAMetaModel() {
        super();
    }

    @Override
    public GenericDataSource createDataSourceImpl(DBRProgressMonitor monitor, DBPDataSourceContainer container) throws DBException {
        return new HANADataSource(monitor, container, this);
    }

    @Override
    public boolean isSchemasOptional() {
        return false;
    }

    @Override
    public List<GenericSchema> loadSchemas(JDBCSession session, GenericDataSource dataSource, GenericCatalog catalog) throws DBException {
        List<GenericSchema> schemas = super.loadSchemas(session, dataSource, catalog);
        // throws exception if password or license expired

        HANASchema publicSchema = new HANASchema(dataSource, catalog, HANAConstants.SCHEMA_PUBLIC);
        int i;
        for (i = 0; i < schemas.size(); i++)
            if (schemas.get(i).getName().compareTo(HANAConstants.SCHEMA_PUBLIC) > 0)
                break;
        schemas.add(i, publicSchema);
        return schemas;
    }

    @Override
    public HANASchema createSchemaImpl(@NotNull GenericDataSource dataSource, @Nullable GenericCatalog catalog, @NotNull String schemaName) {
        return new HANASchema(dataSource, catalog, schemaName);
    }

    @Override
    public GenericTableBase createTableImpl(@NotNull JDBCSession session, @NotNull GenericStructContainer owner, @NotNull GenericMetaObject tableObject, @NotNull JDBCResultSet dbResult) {
        String tableType = GenericUtils.safeGetStringTrimmed(tableObject, dbResult, JDBCConstants.TABLE_TYPE);
        if (tableType != null && tableType.equals("SYNONYM"))
            return null;
        return super.createTableImpl(session, owner, tableObject, dbResult);
    }

    @Override
    public GenericTableBase createTableImpl(GenericStructContainer container,
            @Nullable String tableName, @Nullable String tableType, @Nullable JDBCResultSet dbResult) {
        if (tableType != null && isView(tableType)) {
            return new HANAView(container, tableName, tableType, dbResult);
        }
        return new HANATable(container, tableName, tableType, dbResult);
    }
    
    @Override
    public GenericTableColumn createTableColumnImpl(@NotNull DBRProgressMonitor monitor,
            JDBCResultSet dbResult, @NotNull GenericTableBase table, String columnName, 
            String typeName, int valueType, int sourceType, int ordinalPos, long columnSize, 
            long charLength, Integer scale, Integer precision, int radix, boolean notNull, 
            String remarks, String defaultValue, boolean autoIncrement, boolean autoGenerated) throws DBException {
        if(table.getSchema().getName().equals("SYS") && table.isView()) {
            ((HANADataSource)table.getDataSource()).initializeSysViewColumnUnits(monitor);
            return new HANASysViewColumn(table, columnName,
                    typeName, valueType, sourceType, ordinalPos, columnSize,
                    charLength, scale, precision, radix, notNull,
                    remarks, defaultValue, autoIncrement, autoGenerated);
        } else {
            return new HANATableColumn(table, columnName,
                typeName, valueType, sourceType, ordinalPos, columnSize,
                charLength, scale, precision, radix, notNull,
                remarks, defaultValue, autoIncrement, autoGenerated);
        }
    }
    
    @Override
    public GenericProcedure createProcedureImpl(GenericStructContainer container,
            String procedureName, String specificName, String remarks,
            DBSProcedureType procedureType, GenericFunctionResultType functionResultType) {
        return new HANAProcedure(container, procedureName, specificName, remarks, procedureType, functionResultType);
    }
    
    @Override
    public String getViewDDL(DBRProgressMonitor monitor, GenericView sourceObject, Map<String, Object> options) throws DBException {
        GenericDataSource dataSource = sourceObject.getDataSource();
        try (JDBCSession session = DBUtils.openMetaSession(monitor, sourceObject, "Read HANA view source")) {
            try (JDBCPreparedStatement dbStat = session.prepareStatement(
                "SELECT DEFINITION\n" +
                    "FROM SYS.VIEWS\n" +
                    "WHERE SCHEMA_NAME=? and VIEW_NAME=?"))
            {
                dbStat.setString(1, sourceObject.getContainer().getName());
                dbStat.setString(2, sourceObject.getName());
                try (JDBCResultSet dbResult = dbStat.executeQuery()) {
                    if (dbResult.nextRow()) {
                        return
                            "CREATE VIEW " + sourceObject.getFullyQualifiedName(DBPEvaluationContext.DDL) + " AS\n" +
                            dbResult.getString(1);
                    }
                    return "-- HANA view definition not found";
                }
            }
        } catch (SQLException e) {
            throw new DBException(e, dataSource);
        }
    }

    @Override
    public String getProcedureDDL(DBRProgressMonitor monitor, GenericProcedure sourceObject) throws DBException {
        GenericDataSource dataSource = sourceObject.getDataSource();
        try (JDBCSession session = DBUtils.openMetaSession(monitor, sourceObject, "Read HANA procedure source")) {
            String procedureType = sourceObject.getProcedureType().name();
            try (JDBCPreparedStatement dbStat = session.prepareStatement(
                "SELECT SCHEMA_NAME,"+ procedureType + "_NAME,DEFINITION FROM SYS."+ procedureType + "S\n" +
                    "WHERE SCHEMA_NAME = ? AND " + procedureType + "_NAME = ?"))
            {
                dbStat.setString(1, sourceObject.getContainer().getName());
                dbStat.setString(2, sourceObject.getName());
                try (JDBCResultSet dbResult = dbStat.executeQuery()) {
                    if (dbResult.nextRow()) {
                        return dbResult.getString(3);
                    }
                    return "-- HANA procedure source not found";
                }
            }
        } catch (SQLException e) {
            throw new DBException(e, dataSource);
        }
    }

    @Override
    public String getTableDDL(DBRProgressMonitor monitor, GenericTableBase sourceObject, Map<String, Object> options) throws DBException {
        try (JDBCSession session = DBUtils.openMetaSession(monitor, sourceObject, "Read HANA table DDL")) {
            try (JDBCPreparedStatement dbStat = session.prepareCall(
                "CALL get_object_definition(?,?)"))
            {
                dbStat.setString(1, DBUtils.getQuotedIdentifier(sourceObject.getContainer()));
                dbStat.setString(2, DBUtils.getQuotedIdentifier(sourceObject));
                try (JDBCResultSet dbResult = dbStat.executeQuery()) {
                    StringBuilder ddl = new StringBuilder();
                    while (dbResult.nextRow()) {
                        ddl.append(dbResult.getString("OBJECT_CREATION_STATEMENT"));
                    }
                    if (ddl.length() > 0) {
                        // Format DDL
                        return SQLFormatUtils.formatSQL(sourceObject.getDataSource(), ddl.toString());
                    }
                }
            }
        } catch (Exception e) {
            log.debug("Error reading DDL from HANA server", e);
        }

        return super.getTableDDL(monitor, sourceObject, options);
    }

    @Override
    public boolean supportsTableDDLSplit(GenericTableBase sourceObject) {
        return false;
    }

    @Override
    public boolean supportsTriggers(@NotNull GenericDataSource dataSource) {
        return true;
    }

    @Override
    public JDBCStatement prepareTableTriggersLoadStatement(@NotNull JDBCSession session, @NotNull GenericStructContainer container, @Nullable GenericTableBase table) throws SQLException {
        JDBCPreparedStatement dbStat = session.prepareStatement(
                "SELECT T.SUBJECT_TABLE_NAME AS OWNER, T.* FROM SYS.TRIGGERS T WHERE SUBJECT_TABLE_SCHEMA=?" + (table != null ? " AND SUBJECT_TABLE_NAME=?" : ""));
        dbStat.setString(1, container.getName());
        if (table != null) {
            dbStat.setString(2, table.getName());
        }
        return dbStat;
    }

    @Override
    public GenericTableTrigger createTableTriggerImpl(@NotNull JDBCSession session, @NotNull GenericStructContainer container, @NotNull GenericTableBase genericTableBase, String triggerName, @NotNull JDBCResultSet resultSet) {
        if (CommonUtils.isEmpty(triggerName)) {
            triggerName = JDBCUtils.safeGetString(resultSet, "TRIGGER_NAME");
        }
        return new HANATrigger(genericTableBase, triggerName, resultSet);
    }

    @Override
    public List<? extends GenericTrigger> loadTriggers(DBRProgressMonitor monitor, @NotNull GenericStructContainer container, @Nullable GenericTableBase table) throws DBException {
        if (table == null) {
            return Collections.emptyList();
        }
        try (JDBCSession session = DBUtils.openMetaSession(monitor, container, "Read triggers")) {
            try (JDBCPreparedStatement dbStat = session.prepareStatement("SELECT TRIGGER_NAME FROM SYS.TRIGGERS WHERE SUBJECT_TABLE_SCHEMA=? AND SUBJECT_TABLE_NAME=?")) {
                dbStat.setString(1, table.getSchema().getName());
                dbStat.setString(2, table.getName());
                List<GenericTrigger> result = new ArrayList<>();
                try (JDBCResultSet dbResult = dbStat.executeQuery()) {
                    while (dbResult.next()) {
                        String name = JDBCUtils.safeGetString(dbResult, 1);
                        result.add(new GenericTableTrigger(table, name, null));
                    }
                }
                return result;
            }
        } catch (SQLException e) {
            throw new DBException(e, container.getDataSource());
        }
    }

    @Override
    public String getTriggerDDL(@NotNull DBRProgressMonitor monitor, @NotNull GenericTrigger sourceObject) throws DBException {
        if (sourceObject instanceof HANATrigger) {
            return ((HANATrigger) sourceObject).getDefinition();
        }
        GenericDataSource dataSource = sourceObject.getDataSource();
        try (JDBCSession session = DBUtils.openMetaSession(monitor, sourceObject, "Read HANA trigger source")) {
            try (JDBCPreparedStatement dbStat = session.prepareStatement(
                "SELECT SCHEMA_NAME,TRIGGER_NAME,DEFINITION FROM SYS.TRIGGERS\n" +
                    "WHERE SCHEMA_NAME = ? AND TRIGGER_NAME = ?"))
            {
                dbStat.setString(1, sourceObject.getContainer().getName());
                dbStat.setString(2, sourceObject.getName());
                try (JDBCResultSet dbResult = dbStat.executeQuery()) {
                    if (dbResult.nextRow()) {
                        return dbResult.getString(3);
                    }
                    return "-- HANA trigger source not found";
                }
            }
        } catch (SQLException e) {
            throw new DBException(e, dataSource);
        }
    }

    @Override
    public boolean supportsSequences(@NotNull GenericDataSource dataSource) {
        return true;
    }

    @Override
    public JDBCStatement prepareSequencesLoadStatement(@NotNull JDBCSession session, @NotNull GenericStructContainer container) throws SQLException {
        JDBCPreparedStatement dbStat = session.prepareStatement(
            "SELECT SEQUENCE_NAME, MIN_VALUE, MAX_VALUE, INCREMENT_BY FROM SYS.SEQUENCES WHERE SCHEMA_NAME = ? ORDER BY SEQUENCE_NAME");
        dbStat.setString(1, container.getName());
        return dbStat;
    }

    @Override
    public GenericSequence createSequenceImpl(@NotNull JDBCSession session, @NotNull GenericStructContainer container, @NotNull JDBCResultSet dbResult) {
        String name = JDBCUtils.safeGetString(dbResult, 1);
        if (CommonUtils.isEmpty(name)) {
            return null;
        }
        Number minValue = JDBCUtils.safeGetBigDecimal(dbResult, 2);
        Number maxValue = JDBCUtils.safeGetBigDecimal(dbResult, 3);
        Number incrementBy = JDBCUtils.safeGetBigDecimal(dbResult, 4);
        return new GenericSequence(container, name, "", null, minValue, maxValue, incrementBy);
    }

    @Override
    public boolean supportsSynonyms(@NotNull GenericDataSource dataSource) {
        return true;
    }

    @Override
    public JDBCStatement prepareSynonymsLoadStatement(@NotNull JDBCSession session, @NotNull GenericStructContainer container) throws SQLException {
        JDBCPreparedStatement dbStat = session.prepareStatement(
            "SELECT SYNONYM_NAME, OBJECT_TYPE, OBJECT_SCHEMA, OBJECT_NAME FROM SYS.SYNONYMS WHERE SCHEMA_NAME = ? ORDER BY SYNONYM_NAME");
        dbStat.setString(1, container.getName());
        return dbStat;
    }

    @Override
    public GenericSynonym createSynonymImpl(@NotNull JDBCSession session, @NotNull GenericStructContainer container, @NotNull JDBCResultSet dbResult) {
        String name = JDBCUtils.safeGetString(dbResult, 1);
        if (CommonUtils.isEmpty(name)) {
            return null;
        }
        String targetObjectType = JDBCUtils.safeGetString(dbResult, 2);
        String targetObjectSchema = JDBCUtils.safeGetString(dbResult, 3);
        String targetObjectName = JDBCUtils.safeGetString(dbResult, 4);
        return new HANASynonym(container, name, targetObjectType, targetObjectSchema, targetObjectName);
    }

    @Override
    public JDBCStatement prepareUniqueConstraintsLoadStatement(@NotNull JDBCSession session,
            @NotNull GenericStructContainer owner, @Nullable GenericTableBase forParent) throws SQLException {
        JDBCPreparedStatement dbStat;
        if(forParent!=null) { 
            dbStat = session.prepareStatement("SELECT"
                    + " TABLE_NAME, COLUMN_NAME, POSITION AS KEY_SEQ, CONSTRAINT_NAME AS PK_NAME, IS_PRIMARY_KEY" 
                    + " FROM SYS.CONSTRAINTS"
                    + " WHERE SCHEMA_NAME=? AND TABLE_NAME=?"
                    + " ORDER BY PK_NAME");
            dbStat.setString(1, forParent.getSchema().getName());
            dbStat.setString(2, forParent.getName());
        } else {
            dbStat = session.prepareStatement("SELECT"
                    + " TABLE_NAME, COLUMN_NAME, POSITION AS KEY_SEQ, CONSTRAINT_NAME AS PK_NAME, IS_PRIMARY_KEY" 
                    + " FROM SYS.CONSTRAINTS"
                    + " WHERE SCHEMA_NAME=? "
                    + " ORDER BY TABLE_NAME,PK_NAME");
            dbStat.setString(1, owner.getName());
        }
        return dbStat;
    }

    @Override
    public DBSEntityConstraintType getUniqueConstraintType(JDBCResultSet dbResult) throws DBException, SQLException {
        return JDBCUtils.safeGetBoolean(dbResult, "IS_PRIMARY_KEY", HANAConstants.SYS_BOOLEAN_TRUE)
        		? DBSEntityConstraintType.PRIMARY_KEY : DBSEntityConstraintType.UNIQUE_KEY;
    }

    @Override
    public String getAutoIncrementClause(GenericTableColumn column) {
        return "GENERATED ALWAYS AS IDENTITY";
    }

    @Override
    public boolean isSystemSchema(GenericSchema schema) {
        String schemaName = schema.getName();
        return schemaName.startsWith("_SYS_") ||
            schemaName.startsWith("SAP_") ||
            schemaName.startsWith("HANA_") ||
            schemaName.equals("SYS") ||
            schemaName.equals("SYS_DATABASES");
    }

    @Override
    public boolean isSystemTable(GenericTableBase table) {
        // empty schemas are still shown, so hiding everything in system schemas looks strange
        //if (table.getSchema().getName().startsWith("_SYS_"))
        //    return true;
        return table.getName().startsWith("_SYS_");
    }
    
    @Override
    public DBPErrorAssistant.ErrorPosition getErrorPosition(@NotNull Throwable error) {
        String message = error.getMessage();
        if (!CommonUtils.isEmpty(message)) {
            Matcher matcher = ERROR_POSITION_PATTERN.matcher(message);
            if (matcher.find()) {
                DBPErrorAssistant.ErrorPosition pos = new DBPErrorAssistant.ErrorPosition();
                pos.line = -1;
                pos.position = Integer.parseInt(matcher.group(1)) - 1;
                return pos;
            }
        }
        return null;
    }
}