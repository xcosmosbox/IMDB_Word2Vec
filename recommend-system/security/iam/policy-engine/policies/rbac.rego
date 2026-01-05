package authz

default allow = false

# 允许管理员做任何事
allow {
    input.user.roles[_] == "admin"
}

# 允许用户读取自己的数据
allow {
    input.method == "GET"
    input.path = ["api", "v1", "users", user_id]
    input.user.id == user_id
}

# 允许基于角色的访问
allow {
    role_permissions := data.roles[input.user.roles[_]]
    permission := role_permissions[_]
    permission.method == input.method
    glob.match(permission.path, ["/"], input.path_str)
}

