{% extends 'markdown/index.md.j2' %}

{% block data_png %}
![png](/zachparent-site/{{ output.metadata.filenames['image/png'] | path2url }})
{% endblock data_png %}

{% block data_jpg %}
![jpg](/zachparent-site/{{ output.metadata.filenames['image/jpeg'] | path2url }})
{% endblock data_jpg %}
