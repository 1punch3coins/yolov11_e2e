#ifndef DET_STRUCTS_H_
#define DET_STRUCTS_H_

struct Bbox2D {
    int32_t cls_id;
    std::string cls_name;
    float cls_conf;

    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;

    Bbox2D():
        cls_id(0), cls_conf(0), x(0), y(0), w(0), h(0)
    {}
    Bbox2D(int32_t cls_id_, float cls_conf_, int32_t x_, int32_t y_, int32_t w_, int32_t h_):
        cls_id(cls_id_), cls_conf(cls_conf_), x(x_), y(y_), w(w_), h(h_)
    {}
    Bbox2D(int32_t cls_id_, std::string cls_name_, float cls_conf_, int32_t x_, int32_t y_, int32_t w_, int32_t h_):
        cls_id(cls_id_), cls_name(cls_name_), cls_conf(cls_conf_), x(x_), y(y_), w(w_), h(h_)
    {}
};

#endif
