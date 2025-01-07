package com.example.achats.service;

import com.example.achats.model.Devis;
import com.example.achats.repository.DevisRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DevisService {

    @Autowired
    private DevisRepository devisRepository;

    public List<Devis> findAll() {
        return devisRepository.findAll();
    }

    public Devis save(Devis devis) {
        return devisRepository.save(devis);
    }

    public Devis findById(Long id) {
        return devisRepository.findById(id).orElse(null);
    }

    public void deleteById(Long id) {
        devisRepository.deleteById(id);
    }
}
