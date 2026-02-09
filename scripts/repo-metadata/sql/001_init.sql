-- repo-metadata 仓库元数据 PostgreSQL 初始化
-- 目标: 结构化管理仓库目录/文件的元数据描述

create table if not exists repo_metadata_nodes (
  path         text primary key,                -- 相对路径, e.g. 'src/components'
  type         text not null                     -- 'directory' | 'file'
                 check (type in ('directory', 'file')),
  description  text,                             -- 一句话描述
  detail       text,                             -- 详细说明（可选）
  tags         text[] not null default '{}',     -- 标签数组
  parent_path  text references repo_metadata_nodes(path) on delete cascade,
  sort_order   integer not null default 0,
  updated_by   text not null default 'scan',     -- 'scan' | 'llm' | 'human'
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create index if not exists idx_repo_metadata_nodes_parent
  on repo_metadata_nodes(parent_path, sort_order, path);

-- updated_at 自动更新触发器
create or replace function set_repo_metadata_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_repo_metadata_nodes_updated_at on repo_metadata_nodes;
create trigger trg_repo_metadata_nodes_updated_at
before update on repo_metadata_nodes
for each row
execute function set_repo_metadata_updated_at();
