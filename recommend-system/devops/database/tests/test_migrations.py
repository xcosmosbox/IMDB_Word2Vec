"""
数据库迁移 SQL 文件测试

测试 SQL 迁移文件的语法和结构正确性。
"""

import pytest
import os
import re
from pathlib import Path


class TestMigrationFiles:
    """迁移文件测试类"""
    
    @pytest.fixture
    def migrations_dir(self):
        """获取迁移目录路径"""
        current_dir = Path(__file__).parent.parent
        return current_dir / "migrations"
    
    def test_migrations_directory_exists(self, migrations_dir):
        """测试迁移目录存在"""
        assert migrations_dir.exists(), f"Migrations directory not found: {migrations_dir}"
    
    def test_flyway_config_exists(self, migrations_dir):
        """测试 Flyway 配置文件存在"""
        config_file = migrations_dir / "flyway.conf"
        assert config_file.exists(), "flyway.conf not found"
    
    def test_all_migrations_exist(self, migrations_dir):
        """测试所有必需的迁移文件存在"""
        required_migrations = [
            "V001__initial_schema.sql",
            "V002__add_user_behaviors.sql",
            "V003__add_recommendations.sql",
            "V004__add_indexes.sql",
            "V005__add_vector_extension.sql",
        ]
        
        for migration in required_migrations:
            migration_file = migrations_dir / migration
            assert migration_file.exists(), f"Migration file not found: {migration}"
    
    def test_migration_naming_convention(self, migrations_dir):
        """测试迁移文件命名规范"""
        pattern = re.compile(r"V\d{3}__[a-z_]+\.sql$")
        
        sql_files = list(migrations_dir.glob("V*.sql"))
        
        for sql_file in sql_files:
            assert pattern.match(sql_file.name), \
                f"Invalid migration file name: {sql_file.name}"
    
    def test_migration_version_sequence(self, migrations_dir):
        """测试迁移版本顺序"""
        sql_files = sorted(migrations_dir.glob("V*.sql"))
        
        versions = []
        for sql_file in sql_files:
            match = re.match(r"V(\d{3})__", sql_file.name)
            if match:
                versions.append(int(match.group(1)))
        
        # 检查版本是否连续
        expected = list(range(1, len(versions) + 1))
        assert versions == expected, f"Migration versions are not sequential: {versions}"
    
    def test_migration_files_not_empty(self, migrations_dir):
        """测试迁移文件不为空"""
        sql_files = list(migrations_dir.glob("V*.sql"))
        
        for sql_file in sql_files:
            content = sql_file.read_text(encoding="utf-8")
            assert len(content.strip()) > 0, f"Migration file is empty: {sql_file.name}"
    
    def test_v001_creates_core_tables(self, migrations_dir):
        """测试 V001 创建核心表"""
        migration_file = migrations_dir / "V001__initial_schema.sql"
        content = migration_file.read_text(encoding="utf-8")
        
        # 检查核心表创建语句
        assert "CREATE TABLE users" in content, "users table creation not found"
        assert "CREATE TABLE items" in content, "items table creation not found"
        assert "CREATE TABLE item_stats" in content, "item_stats table creation not found"
    
    def test_v002_creates_behavior_tables(self, migrations_dir):
        """测试 V002 创建行为表"""
        migration_file = migrations_dir / "V002__add_user_behaviors.sql"
        content = migration_file.read_text(encoding="utf-8")
        
        assert "CREATE TABLE user_behaviors" in content, \
            "user_behaviors table creation not found"
        assert "PARTITION BY" in content, \
            "Partition definition not found"
    
    def test_v003_creates_recommendation_tables(self, migrations_dir):
        """测试 V003 创建推荐表"""
        migration_file = migrations_dir / "V003__add_recommendations.sql"
        content = migration_file.read_text(encoding="utf-8")
        
        assert "CREATE TABLE recommendation_requests" in content, \
            "recommendation_requests table creation not found"
        assert "CREATE TABLE recommendation_results" in content, \
            "recommendation_results table creation not found"
        assert "CREATE TABLE experiments" in content, \
            "experiments table creation not found"
    
    def test_v004_creates_indexes(self, migrations_dir):
        """测试 V004 创建索引"""
        migration_file = migrations_dir / "V004__add_indexes.sql"
        content = migration_file.read_text(encoding="utf-8")
        
        assert "CREATE INDEX" in content, "Index creation not found"
        assert "CREATE MATERIALIZED VIEW" in content, \
            "Materialized view creation not found"
    
    def test_v005_adds_vector_extension(self, migrations_dir):
        """测试 V005 添加向量扩展"""
        migration_file = migrations_dir / "V005__add_vector_extension.sql"
        content = migration_file.read_text(encoding="utf-8")
        
        assert "CREATE EXTENSION" in content and "vector" in content, \
            "pgvector extension creation not found"
        assert "CREATE TABLE item_embeddings" in content, \
            "item_embeddings table creation not found"
        assert "CREATE TABLE user_embeddings" in content, \
            "user_embeddings table creation not found"
    
    def test_sql_syntax_basic(self, migrations_dir):
        """测试 SQL 基本语法"""
        sql_files = list(migrations_dir.glob("V*.sql"))
        
        for sql_file in sql_files:
            content = sql_file.read_text(encoding="utf-8")
            
            # 检查是否有未闭合的括号
            open_parens = content.count("(")
            close_parens = content.count(")")
            assert open_parens == close_parens, \
                f"Unbalanced parentheses in {sql_file.name}"
    
    def test_flyway_config_content(self, migrations_dir):
        """测试 Flyway 配置内容"""
        config_file = migrations_dir / "flyway.conf"
        content = config_file.read_text(encoding="utf-8")
        
        # 检查必要的配置项
        assert "flyway.url" in content, "Database URL config not found"
        assert "flyway.locations" in content, "Migration locations not found"
        assert "flyway.encoding" in content or "UTF-8" in content, \
            "Encoding config not found"


class TestBackupScripts:
    """备份脚本测试类"""
    
    @pytest.fixture
    def backup_dir(self):
        """获取备份脚本目录"""
        current_dir = Path(__file__).parent.parent
        return current_dir / "backup"
    
    def test_backup_directory_exists(self, backup_dir):
        """测试备份目录存在"""
        assert backup_dir.exists(), f"Backup directory not found: {backup_dir}"
    
    def test_backup_scripts_exist(self, backup_dir):
        """测试备份脚本文件存在"""
        required_scripts = [
            "backup.sh",
            "restore.sh",
            "verify.sh",
            "cronjob.yaml",
        ]
        
        for script in required_scripts:
            script_file = backup_dir / script
            assert script_file.exists(), f"Script file not found: {script}"
    
    def test_backup_script_has_shebang(self, backup_dir):
        """测试备份脚本有 shebang"""
        shell_scripts = ["backup.sh", "restore.sh", "verify.sh"]
        
        for script in shell_scripts:
            script_file = backup_dir / script
            content = script_file.read_text(encoding="utf-8")
            assert content.startswith("#!/bin/bash"), \
                f"Missing shebang in {script}"
    
    def test_backup_script_has_error_handling(self, backup_dir):
        """测试备份脚本有错误处理"""
        shell_scripts = ["backup.sh", "restore.sh", "verify.sh"]
        
        for script in shell_scripts:
            script_file = backup_dir / script
            content = script_file.read_text(encoding="utf-8")
            assert "set -e" in content or "set -euo pipefail" in content, \
                f"Missing error handling in {script}"
    
    def test_cronjob_yaml_valid(self, backup_dir):
        """测试 CronJob YAML 有效"""
        cronjob_file = backup_dir / "cronjob.yaml"
        content = cronjob_file.read_text(encoding="utf-8")
        
        # 检查必要的 Kubernetes 配置
        assert "apiVersion:" in content, "Missing apiVersion"
        assert "kind: CronJob" in content, "Missing kind: CronJob"
        assert "schedule:" in content, "Missing schedule"


class TestDatabaseScripts:
    """数据库脚本测试类"""
    
    @pytest.fixture
    def scripts_dir(self):
        """获取脚本目录"""
        current_dir = Path(__file__).parent.parent
        return current_dir / "scripts"
    
    def test_scripts_directory_exists(self, scripts_dir):
        """测试脚本目录存在"""
        assert scripts_dir.exists(), f"Scripts directory not found: {scripts_dir}"
    
    def test_required_scripts_exist(self, scripts_dir):
        """测试必需脚本存在"""
        required_scripts = [
            "init-db.sh",
            "seed-data.sh",
            "cleanup.sh",
        ]
        
        for script in required_scripts:
            script_file = scripts_dir / script
            assert script_file.exists(), f"Script not found: {script}"
    
    def test_scripts_have_shebang(self, scripts_dir):
        """测试脚本有 shebang"""
        shell_scripts = scripts_dir.glob("*.sh")
        
        for script_file in shell_scripts:
            content = script_file.read_text(encoding="utf-8")
            assert content.startswith("#!/bin/bash"), \
                f"Missing shebang in {script_file.name}"
    
    def test_init_script_has_required_functions(self, scripts_dir):
        """测试初始化脚本有必需函数"""
        init_script = scripts_dir / "init-db.sh"
        content = init_script.read_text(encoding="utf-8")
        
        required_functions = [
            "create_database",
            "install_extensions",
            "run_migrations",
        ]
        
        for func in required_functions:
            assert func in content, f"Function {func} not found in init-db.sh"
    
    def test_cleanup_script_has_required_functions(self, scripts_dir):
        """测试清理脚本有必需函数"""
        cleanup_script = scripts_dir / "cleanup.sh"
        content = cleanup_script.read_text(encoding="utf-8")
        
        required_functions = [
            "cleanup_behaviors",
            "reclaim_storage",
        ]
        
        for func in required_functions:
            assert func in content, f"Function {func} not found in cleanup.sh"

