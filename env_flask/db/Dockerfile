# compose.yml の image で指定しているバージョンに合わせる
FROM mysql:9.3

# my.cnf をコピーし、所有者とパーミッションを設定
COPY --chown=mysql:mysql my.cnf /etc/mysql/conf.d/my.cnf
RUN chmod 644 /etc/mysql/conf.d/my.cnf