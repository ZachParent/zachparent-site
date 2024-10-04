{% extends 'markdown/index.md.j2' %}

{% block data_png %}
![png](../{{ output.metadata.filenames['image/png'] | path2url }})
{% endblock data_png %}

{% block data_jpg %}
![jpg](../{{ output.metadata.filenames['image/jpeg'] | path2url }})
{% endblock data_jpg %}
