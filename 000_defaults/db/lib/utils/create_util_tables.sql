/* 
    script for creating property utilities tables
    - Simple Feature Access geometry tables for 
        hotel location PostGIS points
    - ref (reference) lookup table for defining and
        using categorical values
    Prerequisits:
    - first the DATABASE property and 
        SCHEMA curated must be created
    - ensure user rezaware has priviledges to drop and
        create tables, views, and so on

    Contributors:
        nuwan.waidyanatha@rezgateway.com

*/
DROP TABLE IF EXISTS curated.util_geom;
CREATE TABLE curated.util_geom
(
  gis_pk serial primary key,
  realm character varying (200),
  realm_fk integer,
  feature character varying (200),
  geom geometry,
  epsg integer,
  geocode character varying (200),
  summary character varying (200),
/*  source_uuid character varying (200), */
/*  data_source character varying(100),  */
  owner_name character varying(200),
  owner_rec_id character varying(100),
/*  data_owner character varying(100),   */
  created_dt timestamp without time zone,
  created_by character varying(100),
  created_proc character varying(100),
  modified_dt timestamp without time zone,
  modified_by character varying(100),
  modified_proc character varying(100),
  deactivate_dt timestamp without time zone
)
WITH (
  OIDS=FALSE
);
comment on column curated.util_geom.gis_pk is 'system generated integer upon insert';
comment on column curated.util_geom.realm is 'realm is the table entity the geom is defined for';
comment on column curated.util_geom.realm_fk is 'realm (entity) table primary key value';
comment on column curated.util_geom.feature is 'categorical value describing the location';
comment on column curated.util_geom.geom is 'point line polygon multipoint coordinates';
comment on column curated.util_geom.epsg is 'epsg value e.g. 4326';
comment on column curated.util_geom.geocode is 'geocode if available';
comment on column curated.util_geom.summary is 'quailifier description about the location';
comment on column curated.util_geom.source_id is 'source url or identifier data was extracted from';
comment on column curated.util_geom.source_type is 'source type e.g. AWS S3, Azure DF, GCS';
/* comment on column curated.util_geom.source_uuid is 'uuid or pk of the record in the data source';*/
/* comment on column curated.util_geom.data_source is 'data storage location name e.g. S3 folder';*/
comment on column curated.util_geom.owner_name is 'name of the data owner originating entity';
comment on column curated.util_geom.owner_rec_id is 'uique record id given by origin data owner';
/* comment on column curated.util_geom.data_owner is 'reference name to the origin of the data'; */

CREATE INDEX idx_gis_geom
ON curated.util_geom (realm, realm_fk, feature);

DROP TABLE IF EXISTS curated.util_refer;
CREATE TABLE curated.util_refer
(
  ref_pk serial primary key,
  realm character varying (100),
  category character varying (200),
  code character varying (100),
  value character varying (200),
  description character varying (200),
/*  source_uuid character varying (200), */
/*  data_source character varying(100),  */
  owner_name character varying(200),
  owner_rec_id character varying(100),
/*  data_owner character varying(100),   */
  created_dt timestamp without time zone,
  created_by character varying(100),
  created_proc character varying(100),
  modified_dt timestamp without time zone,
  modified_by character varying(100),
  modified_proc character varying(100),
  deactivate_dt timestamp without time zone
)
WITH (
  OIDS=FALSE
);
comment on column curated.util_refer.ref_pk is 'system generated integer upon insert';
comment on column curated.util_refer.realm is 'schema table entity the realm is associated with';
comment on column curated.util_refer.category is 'a category within the table the lookup is for';
comment on column curated.util_refer.value is 'the used reference value for the specific category';
comment on column curated.util_refer.code is 'alternate code to the reference value';
comment on column curated.util_refer.description is 'description about the reference value';
comment on column curated.util_refer.source_id is 'source url or identifier data was extracted from';
comment on column curated.util_refer.source_type is 'source type e.g. AWS S3, Azure DF, GCS';
/* comment on column curated.util_refer.source_uuid is 'uuid or pk of the record in the data source';*/
/* comment on column curated.util_refer.data_source is 'data storage location name e.g. S3 folder';*/
comment on column curated.util_refer.owner_name is 'name of the data owner originating entity';
comment on column curated.util_refer.owner_rec_id is 'uique record id given by origin data owner';
/* comment on column curated.util_refer.data_owner is 'reference name to the origin of the data'; */

CREATE INDEX idx_util_refer
ON curated.util_refer (realm, category, value);


ALTER TABLE curated.util_geom
  OWNER TO rezaware;
ALTER TABLE curated.util_refer
  OWNER TO rezaware;